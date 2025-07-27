import os
from dotenv import load_dotenv
import langchain_google_genai as genai
import streamlit as st
from sentence_transformers import SentenceTransformer
import os
import pymongo
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap
import time
import streamlit as st

def safe_log_info(message):
    print(f"INFO: {message}")

def safe_log_warning(message):
    print(f"WARNING: {message}")

def safe_log_error(message, exc_info=False):
    print(f"ERROR: {message}")
    if exc_info:
        import traceback
        traceback.print_exc()
        safe_log_error("Error occurred during logging", exc_info=True)

load_dotenv()
google_api_key = st.secrets['GOOGLE_API_KEY']
mongo_uri = st.secrets['MONGODB_URI']

@st.cache_resource
def load_generative_model():
    llm = ChatGoogleGenerativeAI(
    model = 'models/gemini-2.0-flash',
    temperature=0.2,
    max_tokens = None,
    timeout = 180,
    max_retries = 2,
    convert_system_message_to_human= True,
    api_key = google_api_key
    )
    return llm

@st.cache_resource
def load_embedding_model():
    embedding_model = SentenceTransformer("namdp-ptit/Vidense")
    return embedding_model
@st.cache_resource
def load_mongo_collection():
    client = pymongo.MongoClient(mongo_uri)
    db = client['vietnamese-llms']
    collection = db['vietnamese-llms-data']
    return collection

@st.cache_resource
def load_reranker():
    reranker = CrossEncoder("namdp-ptit/ViRanker")
    return reranker

def get_embedding(text: str) -> list[float]:
    embedding_model = load_embedding_model()
    embedding = embedding_model.encode(text).tolist()
    return embedding

def find_similar_documents_hybrid_search(
    query_vector: List[float],
    search_query: str,
    limit: int = 10,
    candidates: int = 20,
    vector_search_index: str = "embedding_search",
    atlas_search_index: str = "header_text"
) -> List[Dict[str, Any]]:
    """
    Thực hiện tìm kiếm hybrid kết hợp vector search và text search, chạy song song.
    Bao gồm cơ chế fallback nếu tìm kiếm hybrid thất bại.
    """
    all_results = []
    collection = load_mongo_collection()
    # Hàm con cho vector search
    def perform_vector_search() -> list:
        try:
            vector_pipeline = [
                {"$vectorSearch": {
                    "index": vector_search_index,
                    "path": "embedding",
                    "queryVector": query_vector,
                    "limit": limit,
                    "numCandidates": candidates
                }},
                {"$project": {
                    '_id': 1, 'header': 1, 'content': 1, 'uuid': 1,
                    "vector_score": {"$meta": "vectorSearchScore"}
                }}
            ]
            vector_results = list(collection.aggregate(vector_pipeline))
            safe_log_info(f"Vector search trả về {len(vector_results)} kết quả")
            for doc in vector_results:
                doc['combined_score'] = doc.get('vector_score', 0) * 0.6
            return vector_results
        except Exception as e:
            safe_log_warning(f"Vector search thất bại: {e}")
            return []

    # Hàm con cho text search
    def perform_text_search() -> list:
        if not search_query or not search_query.strip():
            return []
        try:
            text_pipeline = [
                {"$search": {
                    "index": atlas_search_index,
                    "text": { # Đơn giản hóa từ compound sang text nếu chỉ có một điều kiện
                        "query": search_query,
                        "path": ["header", "content"] # Thêm keywords vào path
                    }
                }},
                {"$project": {
                    '_id': 1, 'header': 1, 'content': 1, 'uuid': 1, 'keywords': 1,
                    "text_score": {"$meta": "searchScore"}
                }}
            ]
            text_results = list(collection.aggregate(text_pipeline))
            safe_log_info(f"Text search trả về {len(text_results)} kết quả")
            for doc in text_results:
                # Gán trọng số 0.3 cho điểm text search
                doc['combined_score'] = doc.get('text_score', 0) * 0.4
            return text_results
        except Exception as e:
            safe_log_warning(f"Text search thất bại: {e}")
            return []

    try:
        # 1. Chạy song song hai truy vấn
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_search = {
                executor.submit(perform_vector_search): "vector",
                executor.submit(perform_text_search): "text"
            }
            for future in as_completed(future_to_search):
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    safe_log_error(f"Lỗi trong quá trình tìm kiếm song song: {e}")

        search_time = time.time() - start_time
        safe_log_info(f"Tìm kiếm song song hoàn tất trong {search_time:.3f}s")

        # 2. Hợp nhất và loại bỏ trùng lặp (Tối ưu hóa)
        merged_map = {}
        for doc in all_results:
            doc_id = doc['_id']
            if doc_id not in merged_map:
                merged_map[doc_id] = doc
            else:
                # Nếu tài liệu đã tồn tại, cộng dồn điểm số
                merged_map[doc_id]['combined_score'] += doc['combined_score']

        # Chuyển map thành list
        merged_results = list(merged_map.values())

        # 3. Sắp xếp theo điểm số tổng hợp
        merged_results.sort(key=lambda x: x.get('combined_score', 0), reverse=True)

        final_results = merged_results[:limit]
        safe_log_info(f"Tìm kiếm hybrid trả về: {len(final_results)} tài liệu")

        return [{
            '_id': r['_id'],
            'header': r.get('header', ''),
            'content': r.get('content', ''),
            'uuid': r.get('uuid', ''),
            'score': r.get('combined_score', 0)
        } for r in final_results]

    except Exception as e:
        safe_log_error(f"Lỗi nghiêm trọng trong hàm hybrid search: {e}", exc_info=True)

        # ----- PHẦN FALLBACK ĐÃ SỬA -----
        safe_log_warning("Thực hiện fallback: chỉ tìm kiếm bằng Text Search.")
        try:
            # Thực hiện lại một truy vấn text search đơn giản
            fallback_pipeline = [
                {"$search": {
                    "index": atlas_search_index,
                    "text": {
                        "query": search_query,
                        "path": ["header", "content", "keywords"]
                    }
                }},
                {"$project": {
                    '_id': 1, 'header': 1, 'content': 1, 'uuid': 1,
                    'score': {"$meta": "searchScore"}
                }},
                {"$limit": limit}
            ]
            fallback_results = list(collection.aggregate(fallback_pipeline))
            safe_log_info(f"Fallback search trả về {len(fallback_results)} kết quả.")
            return fallback_results
        except Exception as fallback_e:
            safe_log_error(f"Fallback search cũng thất bại: {fallback_e}", exc_info=True)
            return [] # Trả về list rỗng nếu cả fallback cũng lỗi
        
def rerank_documents(query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Reranks a list of documents based on their relevance to the query using a reranker model.

    Args:
        query: The original search query.
        documents: A list of dictionaries, where each dictionary represents a document
                   and contains a 'content' key with the document's text.

    Returns:
        A list of dictionaries representing the reranked documents, sorted by relevance score.
    """
    if not documents:
        return []
    reranker_model = load_reranker()
    # Prepare pairs for the reranker model
    sentence_pairs = [[query, doc.get('content', '')] for doc in documents]

    # Get reranking scores
    rerank_scores = reranker_model.predict(sentence_pairs)

    # Add reranking scores to the documents
    for i, doc in enumerate(documents):
        doc['rerank_score'] = float(rerank_scores[i]) # Convert to float for potential serialization

    # Sort documents by reranking score in descending order
    reranked_documents = sorted(documents, key=lambda x: x.get('rerank_score', -1), reverse=True)

    return reranked_documents

def format_docs(docs):
    return "\n\n".join([doc.get('header', '') + doc.get('content', '') for doc in docs if isinstance(doc, dict) and 'content' in doc and 'header' in doc])

def get_answer_with_rag(query:str) -> str:
    
    revised_template = ChatPromptTemplate.from_messages([
    ('system', """bạn là một trợ lý AI thân thiện, được thiết kế để giúp khám phá mọi điều về Học viện Bưu chính Viễn thông (PTIT).

    QUY TẮC BẮT BUỘC:
    1.  **Quy đổi cơ sở:** Nếu thông tin tham khảo có đề cập đến "cơ sở TP.HCM" hay "cơ sở Hồ Chí Minh", bạn phải tự động quy đổi và trình bày thông tin đó như thể nó thuộc về "cơ sở Hà Đông".
    2.  **Phạm vi trả lời:** Bạn sẽ sử dụng thông tin được cung cấp trong ngữ cảnh để trả lời câu hỏi của người dùng một cách chi tiết và dễ hiểu nhất.
        - Nếu người dùng đặt câu hỏi cụ thể, hãy trả lời dựa vào ngữ cảnh.
        - Nếu câu hỏi nằm ngoài phạm vi thông tin đó, bạn phải nói rõ rằng bạn không có thông tin.
        - Nếu người dùng chỉ giao tiếp xã giao (ví dụ: "xin chào", "bạn là ai?"), hãy phản hồi một cách thân thiện với vai trò là trợ lý PTIT và sẵn sàng giúp đỡ, không trích xuất thông tin từ ngữ cảnh trừ khi được hỏi."""),
    ('human', "Thông tin tham khảo:\n```\n{context}\n```\n\nCâu hỏi của tôi:\n{question}")
    ])
    llm = load_generative_model()
    query_embedding = get_embedding(query)
    
    context_docs = find_similar_documents_hybrid_search(
        query_vector=query_embedding,
        search_query=query,
        limit=10,
        candidates=20, 
        vector_search_index="embedding_search",
        atlas_search_index="header_text"
    )
    
    reranked_docs = rerank_documents(query, context_docs)
    top_n_docs = reranked_docs[:10]
    context = format_docs(top_n_docs)
    
    chain = (
        RunnableMap({
                "context": RunnablePassthrough(),
                "question": RunnablePassthrough()
            })
        | revised_template
        | llm
        | StrOutputParser()
    )
    response = chain.invoke({
        "context": context,
        "question": query})
    return response
