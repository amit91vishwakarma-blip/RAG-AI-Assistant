import requests
import streamlit as st

st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BACKEND_URL = "http://127.0.0.1:8000"

if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = set()

st.markdown(
    """
    <style>

    .block-container {
        max-width: 950px;
        padding-top: 35px;
        padding-bottom: 120px;
    }

    .rag-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 8px;
    }

    .rag-subtitle {
        text-align: center;
        color: #888888;
        font-size: 16px;
        margin-bottom: 45px;
    }

    [data-testid="stChatMessage"] {
        border-radius: 16px;
        margin-bottom: 10px;
    }

    [data-testid="stChatInput"] {
        border-radius: 22px;
    }

    .footer-text {
        text-align: center;
        color: #777777;
        font-size: 12px;
        margin-top: 20px;
    }

    .file-info {
        background: #202127;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 10px;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    '<div class="rag-title">🤖 RAG AI Assistant</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="rag-subtitle">'
    'Ask questions from PDFs, images and websites'
    '</div>',
    unsafe_allow_html=True
)
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    with st.chat_message(role):
        st.markdown(content)
prompt_data = st.chat_input(
    "Ask anything or paste a website URL...",
    accept_file=True,
    file_type=[
        "pdf",
        "png",
        "jpg",
        "jpeg",
        "webp"
    ],
    max_chars=5000
)
if prompt_data is not None:
    question = prompt_data.text.strip()
    uploaded_files = prompt_data.files
    if not question and not uploaded_files:
        st.warning(
            "Please enter a question, URL, or upload a file."
        )
        st.stop()
    user_message = ""
    if question:
        user_message = question
    if uploaded_files:
        if user_message:
            user_message += "\n\n"
        user_message += "📎 "
        user_message += ", ".join(
            file.name
            for file in uploaded_files
        )
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_message
        }
    )
    with st.chat_message("user"):
        st.markdown(user_message)
    file_processing_success = True
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            file_key = (
                file_name,
                len(uploaded_file.getvalue())
            )
            if file_key in st.session_state.uploaded_files:
                continue
            if file_name.lower().endswith(".pdf"):
                with st.spinner(
                    f"Processing {file_name}..."
                ):
                    try:
                        files = {
                            "file": (
                                file_name,
                                uploaded_file.getvalue(),
                                "application/pdf"
                            )
                        }
                        response = requests.post(
                            f"{BACKEND_URL}/ingest/pdf",
                            files=files,
                            timeout=120
                        )
                        if response.status_code == 200:
                            st.session_state.uploaded_files.add(
                                file_key
                            )
                        else:
                            file_processing_success = False
                            st.error(
                                f"PDF processing failed:\n"
                                f"{response.text}"
                            )
                    except requests.exceptions.ConnectionError:
                        file_processing_success = False
                        st.error(
                            "Cannot connect to FastAPI backend.\n\n"
                            "Make sure FastAPI is running on "
                            "http://127.0.0.1:8000"
                        )
                    except requests.exceptions.Timeout:
                        file_processing_success = False
                        st.error(
                            "PDF processing timed out."
                        )
            elif file_name.lower().endswith(
                (
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".webp"
                )
            ):
                with st.spinner(
                    f"Processing {file_name}..."
                ):
                    try:
                        files = {
                            "file": (
                                file_name,
                                uploaded_file.getvalue(),
                                uploaded_file.type
                            )
                        }
                        response = requests.post(
                            f"{BACKEND_URL}/ingest/image",
                            files=files,
                            timeout=120
                        )
                        if response.status_code == 200:
                            st.session_state.uploaded_files.add(
                                file_key
                            )
                        else:
                            file_processing_success = False
                            st.error(
                                f"Image processing failed:\n"
                                f"{response.text}"
                            )
                    except requests.exceptions.ConnectionError:
                        file_processing_success = False
                        st.error(
                            "Cannot connect to FastAPI backend."
                        )
                    except requests.exceptions.Timeout:
                        file_processing_success = False
                        st.error(
                            "Image processing timed out."
                        )

    if (
        question
        and question.lower().startswith(
            ("http://", "https://")
        )
    ):
        with st.spinner(
            "Reading website..."
        ):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/ingest/url",
                    json={
                        "url": question
                    },
                    timeout=120
                )
                if response.status_code == 200:
                    answer = (
                        "🌐 Website processed successfully. "
                        "You can now ask questions about it."
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                    st.rerun()
                else:
                    st.error(
                        response.text
                    )
                    st.stop()
            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to FastAPI backend."
                )
                st.stop()
            except requests.exceptions.Timeout:
                st.error(
                    "Website processing timed out."
                )
                st.stop()
    if (
        question
        and not question.lower().startswith(
            ("http://", "https://")
        )
        and file_processing_success
    ):
        with st.chat_message("assistant"):
            with st.spinner(
                "Thinking..."
            ):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/chat",
                        json={
                            "question": question
                        },
                        timeout=120
                    )

                    if response.status_code == 200:
                        data = response.json()
                        answer = data.get(
                            "answer",
                            "I could not find an answer."
                        )

                        st.markdown(answer)
        
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer
                            }
                        )
                    else:
                        error_message = (
                            "Unable to generate answer."
                        )
                        st.error(
                            f"{error_message}\n\n"
                            f"{response.text}"
                        )
                except requests.exceptions.ConnectionError:
                    st.error(
                        "Cannot connect to FastAPI backend.\n\n"
                        "Start FastAPI first."
                    )
                except requests.exceptions.Timeout:
                    st.error(
                        "The request timed out."
                    )
st.markdown(
    '<div class="footer-text">'
    'RAG AI Assistant • PDF • Image • Website'
    '</div>',
    unsafe_allow_html=True
)