import streamlit as st
from rag import get_answer_with_rag, load_generative_model, load_reranker, load_embedding_model, load_mongo_collection

load_generative_model()
load_reranker()
load_embedding_model()
load_mongo_collection()
st.set_page_config(
    page_title="PTIT RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# --- GIAO DIỆN CHÍNH ---
st.title("🤖 PTIT RAG Chatbot")
st.caption("Trợ lý ảo thông minh về Học viện Bưu chính Viễn thông")

# Khởi tạo session state để lưu trữ lịch sử trò chuyện
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Xin chào! Tôi có thể giúp gì cho bạn về các thông tin tại PTIT?"}
    ]

# --- SIDEBAR ---
with st.sidebar:
    st.header("Tùy chọn")
    if st.button("🗑️ Xóa cuộc trò chuyện", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Cuộc trò chuyện đã được xóa. Hãy bắt đầu lại nhé!"}
        ]
        st.rerun()

    st.markdown("---")
    st.markdown("### Về ứng dụng")
    st.info("Ứng dụng này sử dụng RAG để trả lời câu hỏi dựa trên tài liệu về PTIT.")

# Hiển thị lịch sử trò chuyện
for message in st.session_state.messages:
    avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.write(message["content"])

def submit_question(question: str):
    st.session_state.messages.append({"role": "user", "content": question})
    st.rerun()

# Khu vực nhập liệu của người dùng
if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    # Thêm tin nhắn của người dùng vào session state và hiển thị ngay
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.write(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🤖 Tôi đang suy nghĩ, bạn chờ chút nhé..."):
            try:
                response = get_answer_with_rag(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_message = "Rất tiếc, đã có lỗi xảy ra. Vui lòng thử lại sau!"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
    