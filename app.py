import os
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

# 1. Cloud Fallback & LLM Engine Orchestrator
try:
    from groq import Groq
    groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
    groq_client = Groq(api_key=groq_api_key) if groq_api_key else None
except Exception:
    groq_client = None

def query_rag_llm(system_instruction: str, user_prompt: str) -> str:
    """Attempts local Ollama first; seamlessly switches to Groq Llama 3 on Streamlit Cloud."""
    # Attempt 1: Local Ollama
    try:
        import ollama
        res = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            options={"temperature": 0.1}
        )
        return res["message"]["content"]
    except Exception:
        pass

    # Attempt 2: Groq Cloud Inference
    if groq_client and groq_client.api_key:
        try:
            completion = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Cloud Inference Error: {e}"

    return "Error: Local Ollama is unreachable and no valid GROQ_API_KEY was found in Streamlit Secrets."

# 2. Page Configuration
st.set_page_config(page_title="Helix Local RAG Engine", page_icon="🧬", layout="wide")

# 3. Theme-Adaptive Styling
st.markdown("""
<style>
[data-theme="light"] {
    color: #0f172a !important;
}
[data-theme="light"] p, [data-theme="light"] span, [data-theme="light"] label, [data-theme="light"] div {
    color: #0f172a !important;
}
[data-theme="light"] input, [data-theme="light"] textarea, [data-theme="light"] [data-testid="stChatInputtextarea"] {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 2px solid #cbd5e1 !important;
}
[data-theme="light"] input::placeholder, [data-theme="light"] textarea::placeholder {
    color: #64748b !important;
}
[data-theme="light"] .stButton button {
    background-color: #e2e8f0 !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
}
[data-theme="dark"] {
    color: #f1f5f9 !important;
}
[data-theme="dark"] p, [data-theme="dark"] span, [data-theme="dark"] label {
    color: #f1f5f9 !important;
}
[data-theme="dark"] input, [data-theme="dark"] textarea, [data-theme="dark"] [data-testid="stChatInputtextarea"] {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 2px solid #6366f1 !important;
}
[data-testid="stChatInputtextarea"] p {
    color: #000000 !important;
}
.glow-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.05em;
    margin-bottom: 5px;
}
</style>
""", unsafe_allow_html=True)

# 4. Header Section
st.markdown('<h1 class="glow-title">🧬 Helix Premium RAG Core</h1>', unsafe_allow_html=True)
st.write("An enterprise-grade document intelligence system with precise citation and provenance tracking.")

# 5. Initialize In-Memory Vector Store for Cloud Compatibility
@st.cache_resource
def get_vector_db():
    client = chromadb.Client()
    ef = embedding_functions.DefaultEmbeddingFunction()
    col = client.get_or_create_collection(name="helix_secure_collection", embedding_function=ef)
    return client, col

chroma_client, collection = get_vector_db()

# 6. Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

def chunk_text(text, chunk_size=700, overlap=100):
    """Split text into overlapping chunks on whitespace boundaries."""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space
        chunk = text[start:end].strip()
        if len(chunk) > 30:
            chunks.append(chunk)
        start = end - overlap if end - overlap > start else end
    return chunks

# 7. Sidebar Controls
st.sidebar.markdown("### 📁 Control Dashboard")

if st.sidebar.button("🧹 Clear Conversation History", use_container_width=True):
    st.session_state.messages = []
    st.toast("Chat history wiped successfully!", icon="🧼")
    st.rerun()

uploaded_file = st.sidebar.file_uploader("Ingest target document:", type=["pdf"])

total_chars = 0
total_pages = 0
total_chunks = 0

if uploaded_file is not None:
    if "last_processed_file" not in st.session_state or st.session_state.last_processed_file != uploaded_file.name:
        with st.sidebar.spinner("🧠 Fracturing document vectors..."):
            try:
                chroma_client.delete_collection("helix_secure_collection")
            except Exception:
                pass
            
            collection = chroma_client.get_or_create_collection(
                name="helix_secure_collection", 
                embedding_function=embedding_functions.DefaultEmbeddingFunction()
            )

            reader = PdfReader(uploaded_file)
            total_pages = len(reader.pages)

            documents_to_index = []
            generated_ids = []
            chunk_counter = 0

            for page_index, page in enumerate(reader.pages):
                extracted_text = page.extract_text()
                if extracted_text:
                    total_chars += len(extracted_text)
                    clean_text = extracted_text.replace("\n", " ")
                    for chunk in chunk_text(clean_text):
                        documents_to_index.append(chunk)
                        generated_ids.append(f"Page_{page_index + 1}_Chunk_{chunk_counter}")
                        chunk_counter += 1

            total_chunks = len(documents_to_index)

            if documents_to_index:
                collection.add(documents=documents_to_index, ids=generated_ids)
                st.sidebar.success(f"Loaded: {uploaded_file.name}")
                st.session_state.meta_pages = total_pages
                st.session_state.meta_chars = total_chars
                st.session_state.meta_chunks = total_chunks
                st.session_state.last_processed_file = uploaded_file.name
            else:
                st.sidebar.error("No extractable structural text data found.")

if "last_processed_file" in st.session_state:
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 📊 Document Statistics")
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Pages", st.session_state.get("meta_pages", 0))
    col2.metric("Data Chunks", st.session_state.get("meta_chunks", 0))
    st.sidebar.metric("Total Extracted Characters", f"{st.session_state.get('meta_chars', 0):,}")

# 8. Render Message History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📍 View Verification Provenance Chunks"):
                for text, source_id in msg["sources"]:
                    st.markdown(f"**Source Target Reference ID:** `{source_id}`")
                    st.caption(f"Text Segment Match: *\"{text}\"*")
                    st.markdown("---")

# 9. Query & Retrieval Loop
if user_input := st.chat_input("Query ingested context structures..."):
    if "last_processed_file" not in st.session_state:
        with st.chat_message("assistant"):
            st.warning("Operational Error: Please upload a PDF target context document via the control dashboard panel first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Searching context database and synthesizing response..."):
                search_results = collection.query(query_texts=[user_input], n_results=4)
                retrieved_chunks_list = search_results['documents'][0] if search_results['documents'] else []
                source_reference_keys = search_results['ids'][0] if search_results['ids'] else []

                compiled_context = "\n\n".join(retrieved_chunks_list)

                system_instruction = (
                    "You are an elite technical documentation analyst. Answer user questions "
                    "using ONLY the provided context chunks. Do not assume or extrapolate outside facts. "
                    "Be comprehensive, structured, and factual."
                )
                complete_prompt = f"Context Material Chunks:\n{compiled_context}\n\nQuestion Target: {user_input}"

                ai_answer_string = query_rag_llm(system_instruction, complete_prompt)

            st.markdown(ai_answer_string)

            if retrieved_chunks_list:
                with st.expander("📍 View Verification Provenance Chunks"):
                    for text, source_id in zip(retrieved_chunks_list, source_reference_keys):
                        st.markdown(f"**Source Target Reference ID:** `{source_id}`")
                        st.caption(f"Text Segment Match: *\"{text}\"*")
                        st.markdown("---")

        st.session_state.messages.append({
            "role": "assistant",
            "content": ai_answer_string,
            "sources": list(zip(retrieved_chunks_list, source_reference_keys))
        })