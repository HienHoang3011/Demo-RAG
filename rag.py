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
google_api_key = os.getenv("GOOGLE_API_KEY")
mongo_uri = os.getenv("MONGODB_URI")

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
    query_vector: list[float],
    search_query: str,
    limit: int = 10,
    candidates: int = 20,
    vector_search_index: str = "embedding_search", 
    atlas_search_index: str = "header_text"
) -> list[dict]:
    """
    Hybrid search combining vector and text search with parallel execution.
    """
    all_results = []
    collection = load_mongo_collection()
    def perform_vector_search():
        """Perform vector search in parallel."""
        try:
            vector_pipeline = [
                {
                    "$vectorSearch": {
                        "index": vector_search_index,
                        "path": "embedding",
                        "queryVector": query_vector,
                        "limit": limit,
                        "numCandidates": candidates
                    }
                },
                {
                    "$project": {
                        '_id': 1,
                        'header' : 1,
                        'content': 1,
                        "vector_score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]

            vector_results = list(collection.aggregate(vector_pipeline))
            safe_log_info(f"Vector search returned {len(vector_results)} results")
            for doc in vector_results:
                doc['search_type'] = 'vector'
                doc['combined_score'] = doc.get('vector_score', 0) * 0.6  # Weight vector score
            return vector_results
        except Exception as e:
           safe_log_warning(f"Vector search failed: {e}")
           return []

    def perform_text_search():
        """Perform text search in parallel."""
        if not search_query or not search_query.strip():
            return []

        try:
            text_pipeline = [
                {
                    "$search": {
                        "index": atlas_search_index,
                        "compound": {
                            "must": [
                                {
                                    "text": {
                                        "query": search_query,
                                        "path": ["header", "content"]
                                    }
                                }
                            ]
                        }
                    }
                },
                {
                    "$project": {
                        '_id': 1,
                        'header': 1,
                        'content': 1,
                        "text_score": {"$meta": "searchScore"}
                    }
                }
            ]

            text_results = list(collection.aggregate(text_pipeline))
            safe_log_info(f"Text search returned {len(text_results)} results")
            for doc in text_results:
                doc['search_type'] = 'text'
                doc['combined_score'] = doc.get('text_score', 0) * 0.4  # Weight text score
            return text_results
        except Exception as e:
            safe_log_warning(f"Text search failed: {e}")
            return []

    try:
        # Run both searches in parallel
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=2) as executor:
            vector_future = executor.submit(perform_vector_search)
            text_future = executor.submit(perform_text_search)

            # Collect results as they complete
            for future in as_completed([vector_future, text_future]):
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    safe_log_error(f"Error in parallel search: {e}")

        search_time = time.time() - start_time
        safe_log_info(f"Parallel search completed in {search_time:.3f}s")

        # 3. Merge và deduplicate results
        seen_ids = set()
        merged_results = []

        for doc in all_results:
            doc_id = str(doc['_id'])
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                # Clean up the document for final result
                final_doc = {
                    '_id': doc['_id'],
                    'content': doc.get('content', ''),
                    # 'uploader_username': doc.get('uploader_username', ''), # Removed
                    'header': doc.get('header', ''),
                    'score': doc.get('combined_score', 0)
                }
                merged_results.append(final_doc)
            else:
                # If document already exists, boost its score
                for existing_doc in merged_results:
                    if str(existing_doc['_id']) == doc_id:
                        existing_doc['score'] += doc.get('combined_score', 0) * 0.5
                        break

        # Sort by combined score
        merged_results.sort(key=lambda x: x.get('score', 0), reverse=True)

        # Return top results
        final_results = merged_results[:limit]
        safe_log_info(f"Hybrid search final results: {len(final_results)} documents")

        return final_results

    except Exception as e:
        safe_log_error(f"Error in hybrid search: {e}", exc_info=True)

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
        Bạn sẽ sử dụng thông tin được cung cấp để trả lời các câu hỏi của người dùng một cách chi tiết và dễ hiểu nhất.
        Hãy nhớ rằng, bạn chỉ có thể trả lời dựa trên thông tin bạn cung cấp. Nếu câu hỏi nằm ngoài phạm vi thông tin đó, bạn sẽ cho người dùng biết."""),
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