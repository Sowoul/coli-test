# app.py
import os
import base64
from io import BytesIO
from dotenv import load_dotenv

import streamlit as st
from huggingface_hub import login
from byaldi import RAGMultiModalModel
from PIL import Image
from openai import OpenAI

# ========== Config & Setup ==========
load_dotenv()
st.set_page_config(
    layout="wide", 
    page_title="Multimodal RAG",
    page_icon="🤖",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    
    .stChatMessage {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .chat-input {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        padding: 1rem;
        border-top: 1px solid #e0e0e0;
        z-index: 999;
    }
    
    .upload-area {
        border: 2px dashed #d0d0d0;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #4f46e5;
        background-color: #f8fafc;
    }
    
    .settings-drawer {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-ready { background-color: #10b981; }
    .status-processing { background-color: #f59e0b; }
    .status-error { background-color: #ef4444; }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Make sidebar collapsible */
    .css-1d391kg {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DEFAULT_UPLOAD_DIR = "./uploads"
os.makedirs(DEFAULT_UPLOAD_DIR, exist_ok=True)

# ========== Session State ==========
if 'rag_model' not in st.session_state:
    st.session_state.rag_model = None
if 'document_indexed' not in st.session_state:
    st.session_state.document_indexed = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_document' not in st.session_state:
    st.session_state.current_document = None

# ========== Utils ==========
def save_uploaded_file(uploaded_file, upload_dir=DEFAULT_UPLOAD_DIR):
    save_path = os.path.join(upload_dir, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path

@st.cache_resource
def load_rag_model(model_name: str):
    return RAGMultiModalModel.from_pretrained(model_name, verbose=10)

def index_document(rag, path: str, index_name="doc_index"):
    rag.index(
        input_path=path,
        index_name=index_name,
        store_collection_with_index=True,
        overwrite=True,
    )

def run_rag_search(rag, query: str, k: int = 1):
    return rag.search(query, k=k, return_base64_results=True)

def call_llm(client, model: str, query: str, image_b64: str):
    return client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            }
        ],
        max_tokens=300,
    )

# ========== Main UI ==========
st.markdown("# 🤖 Multimodal RAG Assistant")
st.markdown("*Ask questions about your documents using advanced AI*")

# Settings in collapsible expander
with st.expander("⚙️ Configuration", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        colpali_model = st.selectbox(
            "Retrieval Model",
            options=["vidore/colpali", "vidore/colpali-v1.2", "vidore/colpali-v1.3"],
            index=2,
            help="Choose the ColPali model for document retrieval"
        )
    
    with col2:
        multi_model_llm = st.selectbox(
            "Language Model",
            options=["gpt-4o-mini", "gpt-4o", "gpt-4-vision-preview"],
            index=0,
            help="Select the LLM for generating responses"
        )

# Document Upload Section
with st.expander("📄 Document Management", expanded=not st.session_state.document_indexed):
    uploaded_file = st.file_uploader(
        "Upload a document to analyze",
        type=["pdf", "png", "jpg", "jpeg"],
        help="Supports PDF documents and images"
    )
    
    if uploaded_file and uploaded_file != st.session_state.current_document:
        with st.spinner("Processing document..."):
            # Save the file
            save_path = save_uploaded_file(uploaded_file)
            
            # Load model if not already loaded
            if st.session_state.rag_model is None:
                st.session_state.rag_model = load_rag_model(colpali_model)
            
            # Index the document
            index_document(st.session_state.rag_model, save_path)
            
            st.session_state.document_indexed = True
            st.session_state.current_document = uploaded_file
            st.session_state.chat_history = []  # Clear chat history for new document
            
        st.success(f"✅ Document '{uploaded_file.name}' has been processed and indexed!")
        st.rerun()

# Status indicator
if st.session_state.document_indexed:
    st.markdown(
        f'<div style="margin: 1rem 0;"><span class="status-indicator status-ready"></span>'
        f'Ready to answer questions about: <strong>{st.session_state.current_document.name}</strong></div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f'<div style="margin: 1rem 0;"><span class="status-indicator status-error"></span>'
        f'No document loaded. Please upload a document to begin.</div>',
        unsafe_allow_html=True
    )

# Chat History
if st.session_state.chat_history:
    st.markdown("---")
    for i, (query, response, image) in enumerate(st.session_state.chat_history):
        # User message
        with st.container():
            st.markdown(f"""
            <div style="display: flex; justify-content: flex-end; margin: 1rem 0;">
                <div style="background: #4f46e5; color: white; padding: 0.75rem 1rem; border-radius: 18px; max-width: 70%;">
                    {query}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Assistant response
        with st.container():
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                <div style="background: #f1f5f9; padding: 1rem; border-radius: 12px; margin: 0.5rem 0;">
                    {response}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if image:
                    st.image(image, caption="Retrieved Context", use_column_width=True)

# Query Input
st.markdown("---")
query = st.text_input(
    "Ask a question about your document",
    placeholder="What would you like to know about this document?",
    disabled=not st.session_state.document_indexed,
    key="query_input"
)

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    search_clicked = st.button("🔍 Ask", disabled=not st.session_state.document_indexed or not query)
with col2:
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# Process Query
if search_clicked and query and st.session_state.document_indexed:
    with st.spinner("Searching document and generating response..."):
        try:
            # Search for relevant content
            results = run_rag_search(st.session_state.rag_model, query)
            image_b64 = results[0].base64
            image = Image.open(BytesIO(base64.b64decode(image_b64)))
            
            # Get LLM response
            response = call_llm(client, multi_model_llm, query, image_b64)
            output = response.choices[0].message.content
            
            # Add to chat history
            st.session_state.chat_history.append((query, output, image))
            
            # Clear the input
            st.session_state.query_input = ""
            
            st.rerun()
            
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")

# Footer
st.markdown("""
---
<div style="text-align: center; color: #666; font-size: 0.8rem; padding: 2rem 0;">
    Powered by ColPali + OpenAI • Built with Streamlit
</div>
""", unsafe_allow_html=True)