import os
import time
import streamlit as st
import pypdf
import chromadb
from chromadb.utils import embedding_functions

st.set_page_config(page_title="Helix Local RAG Engine", page_icon="🧬", layout="wide")

# ==========================================
# GROQ & INFERENCE UTILITIES
# ==========================================

def get_groq_client():
    """Retrieves Groq client safely from Streamlit Secrets or Environment."""
    api_key = None
    try:
        if "GROQ_API_KEY" in st.secrets:
            api_key = str(st.secrets["GROQ_API_KEY"]).strip()
        elif hasattr(st.secrets, "get") and st.secrets.get("GROQ_API_KEY"):
            api_key = str(st.secrets.get("GROQ_API_KEY")).strip()
    except Exception:
        pass

    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY", "").strip()

    if not api_key:
        return None, "GROQ_API_KEY was not found in Streamlit Secrets or Environment."

    try:
        from groq import Groq
        return Groq(api_key=api_key), None
    except Exception as e:
        return None, f"Groq initialization error: {e}"


def run_llm_inference(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
    """Queries local Ollama if running; otherwise executes via Groq Cloud fallback."""
    # 1. Local Ollama Attempt
    try:
        import ollama
        res = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={"temperature": temperature}
        )
        if res and "message" in res and "content" in res["message"]:
            return res["message"]["content"].strip()
    except Exception:
        pass

    # 2. Groq Cloud Fallback
    client, err = get_groq_client()
    if err:
        return f"Configuration Error: {err}"

    candidate_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "llama3-70b-8192",
        "llama3-8b-8192"
    ]

    try:
        active_models = [
            m.id for m in client.models.list().data
            if not any(x in m.id for x in ["whisper", "guard", "vision", "embed"])
        ]
        if active_models:
            candidate_models = active_models + candidate_models
    except Exception:
        pass

    last_error = ""
    for model_id in candidate_models:
        try:
            completion = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            last_error = str(e)
            continue

    return f"Cloud Inference Error: {last_error}"


# ==========================================
# CHROMADB CACHED CLIENT INITIALIZATION
# ==========================================

@st.cache_resource
def get_chroma_client():
    """Initializes a persistent ChromaDB instance that survives Streamlit script reruns."""
    return chromadb.Client()

chroma_client = get_chroma_client()
embed_fn = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_or_create_collection(
    name="rag_knowledge_base",
    embedding_function=embed_fn
)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100):
    """Splits ingested PDF strings into overlapping semantic chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks


# ==========================================
# STREAMLIT UI
# ==========================================

st.title("🧬 Helix Premium RAG Core")
st.markdown("Local Document Intelligence framework with semantic vector search, chunk provenance citation, and fallback inference.")

# Session state initialization for conversation history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("📁 Control Dashboard")
    
    if st.button("🧹 Clear Conversation History"):
        st.session_state.chat_history = []
        st.rerun()

    st.subheader("Ingest target document:")
    uploaded_file = st.file_uploader("Upload a PDF to parse and embed:", type=["pdf"])

    total_pages = 0
    total_chars = 0
    total_chunks = 0

    if uploaded_file:
        try:
            reader = pypdf.PdfReader(uploaded_file)
            full_pdf_text = ""
            for page_idx, page in enumerate(reader.pages):
                extracted = page.extract_text()
                if extracted:
                    full_pdf_text += f"\n[Page {page_idx + 1}]\n" + extracted
            
            total_pages = len(reader.pages)
            total_chars = len(full_pdf_text)

            chunks = chunk_text(full_pdf_text)
            total_chunks = len(chunks)

            # Store in ChromaDB vector collection
            existing_count = collection.count()
            if existing_count == 0 or st.session_state.get("last_uploaded") != uploaded_file.name:
                # Reset collection to avoid mixing documents
                chroma_client.delete_collection("rag_knowledge_base")
                collection = chroma_client.get_or_create_collection(
                    name="rag_knowledge_base",
                    embedding_function=embed_fn
                )
                
                doc_ids = [f"chunk_{i}" for i in range(len(chunks))]
                doc_metas = [{"source": uploaded_file.name, "chunk_id": i} for i in range(len(chunks))]
                
                collection.add(
                    documents=chunks,
                    metadatas=doc_metas,
                    ids=doc_ids
                )
                st.session_state["last_uploaded"] = uploaded_file.name
                st.success(f"Indexed {total_chunks} chunks successfully!")

        except Exception as e:
            st.error(f"Ingestion Error: {e}")

    st.markdown("---")
    st.subheader("📊 Document Statistics")
    col_a, col_b = st.columns(2)
    col_a.metric("Pages", total_pages)
    col_b.metric("Data Chunks", total_chunks)
    st.metric("Total Extracted Characters", f"{total_chars:,}")

# Render chat history
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(chat["question"])
    with st.chat_message("assistant"):
        st.markdown(chat["answer"])
        if chat.get("sources"):
            with st.expander("🔍 Verified Context Proof & Source Chunks"):
                for s in chat["sources"]:
                    st.markdown(f"> {s}")

# Handle new user queries
user_input = st.chat_input("Query ingested context structures...")

if user_input:
    # Display user question
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Executing vector retrieval & synthesis..."):
            retrieved_chunks = []
            
            # Vector database query
            if collection.count() > 0:
                n_results = min(4, collection.count())
                results = collection.query(query_texts=[user_input], n_results=n_results)
                if results and "documents" in results and results["documents"]:
                    retrieved_chunks = results["documents"][0]

            if retrieved_chunks:
                context_str = "\n\n---\n\n".join(retrieved_chunks)
                system_prompt = (
                    "You are a precise, document-grounded intelligence assistant. "
                    "Answer the user's question using ONLY the retrieved context below. "
                    "Cite exact facts, figures, and technical specifications where available. "
                    "If the answer cannot be deduced strictly from the context, state that the context lacks this information."
                )
                user_prompt = f"Retrieved Context:\n{context_str}\n\nUser Question:\n{user_input}"
                response_text = run_llm_inference(system_prompt, user_prompt)
            else:
                system_prompt = "You are a helpful knowledge assistant. Note: No document was provided for vector search."
                response_text = run_llm_inference(system_prompt, user_input)

            st.markdown(response_text)

            if retrieved_chunks:
                with st.expander("🔍 Verified Context Proof & Source Chunks"):
                    for s in retrieved_chunks:
                        st.markdown(f"> {s}")

    # Persist in session history
    st.session_state.chat_history.append({
        "question": user_input,
        "answer": response_text,
        "sources": retrieved_chunks
    })