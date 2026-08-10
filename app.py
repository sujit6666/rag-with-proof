import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import ollama
from pypdf import PdfReader

# 1. Wide Layout Configuration
st.set_page_config(page_title="Helix Local RAG Engine", page_icon="🧬", layout="wide")

# 2. Inject Adaptive CSS Styles that fix light mode text visibility perfectly
st.markdown("""
    <style>
    /* 🌓 THEME-ADAPTIVE OVERRIDES */
    
    /* When user switches to Streamlit Light Theme */
    [data-theme="light"] {
        color: #0f172a !important;
    }
    [data-theme="light"] p, [data-theme="light"] span, [data-theme="light"] label, [data-theme="light"] div {
        color: #0f172a !important;
    }
    /* Fix light mode input text and placeholder readability */
    [data-theme="light"] input, [data-theme="light"] textarea, [data-theme="light"] [data-testid="stChatInputtextarea"] {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 2px solid #cbd5e1 !important;
    }
    [data-theme="light"] input::placeholder, [data-theme="light"] textarea::placeholder {
        color: #64748b !important;
    }
    /* Fix sidebar buttons in light mode */
    [data-theme="light"] .stButton button {
        background-color: #e2e8f0 !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
    }

    /* When user switches to Streamlit Dark Theme (Keeping your perfect dark layout intact) */
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

    /* Common Chat Input adjustments */
    [data-testid="stChatInputtextarea"] p {
        color: #000000 !important;
    }

    /* Elegant Premium Glowing Header Headline Text */
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

# 3. Render Header Branding Layout Elements
st.markdown('<h1 class="glow-title">🧬 Helix Premium RAG Core</h1>', unsafe_allow_html=True)
st.write("An enterprise-grade document intelligence system running completely local and offline.")

# 4. Initialize Local DB Connections Safely
chroma_client = chromadb.PersistentClient(path="./my_vector_db")
default_ef = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_or_create_collection(name="helix_secure_collection", embedding_function=default_ef)

# 5. Core Chat State Management Cache Logic
if "messages" not in st.session_state:
    st.session_state.messages = []

# 6. Build the Sidebar Management Interface
st.sidebar.markdown("### 📁 Control Dashboard")

# Option 1 Implementation: Clear Chat Button Event Handler Action
if st.sidebar.button("🧹 Clear Conversation History", use_container_width=True):
    st.session_state.messages = []
    st.toast("Chat history wiped successfully!", icon="🧼")
    st.rerun()

uploaded_file = st.sidebar.file_uploader("Ingest target document:", type=["pdf"])

# Variables to hold document layout metrics dashboard tracking numbers
total_chars = 0
total_pages = 0
total_chunks = 0

if uploaded_file is not None:
    if "last_processed_file" not in st.session_state or st.session_state.last_processed_file != uploaded_file.name:
        with st.sidebar.spinner("🧠 Fracturing document vectors..."):
            try:
                chroma_client.delete_collection("helix_secure_collection")
                collection = chroma_client.get_or_create_collection(name="helix_secure_collection", embedding_function=default_ef)
            except Exception:
                pass

            reader = PdfReader(uploaded_file)
            total_pages = len(reader.pages)
            documents_to_index = []
            generated_ids = []
            chunk_counter = 0

            for page_index, page in enumerate(reader.pages):
                extracted_text = page.extract_text()
                if extracted_text:
                    total_chars += len(extracted_text)
                    paragraphs = extracted_text.split("\n\n")
                    for para in paragraphs:
                        clean_para = para.strip().replace("\n", " ")
                        if len(clean_para) > 30:
                            documents_to_index.append(clean_para)
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

# Render Visual Side Panel Analytics Cards
if "last_processed_file" in st.session_state:
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 📊 Document Statistics")
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Pages", st.session_state.get("meta_pages", 0))
    col2.metric("Data Chunks", st.session_state.get("meta_chunks", 0))
    st.sidebar.metric("Total Extracted Characters", f"{st.session_state.get('meta_chars', 0):,}")

# 7. Render Historical Chat Message Dialogue Stream
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 8. Core User Query Action Execution Loop
if user_input := st.chat_input("Query ingested context structures..."):
    if "last_processed_file" not in st.session_state:
        with st.chat_message("assistant"):
            st.warning("Operational Error: Please upload a PDF target context document via the control dashboard panel first.")
    else:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Searching local context database and assembling analytical narrative..."):
                
                # Option 2 Implementation: n_results=4 pulls back paragraphs for complete data lookups
                search_results = collection.query(query_texts=[user_input], n_results=4)
                
                retrieved_chunks_list = search_results['documents'] [0]
                source_reference_keys = search_results['ids'] [0]
                
                compiled_context_paragraph = "\n\n".join(retrieved_chunks_list)

                system_instruction = (
                    "You are an elite, highly accurate technical documentation analyst. Your task is to resolve user questions "
                    "using ONLY the provided text block. Do not use outside facts. Do not invent details. Be comprehensive, detailed, "
                    "and professional."
                )
                complete_prompt = f"Context Material Chunks:\n{compiled_context_paragraph}\n\nQuestion Target: {user_input}"

                # Trigger local LLM compilation model engine execution loops offline
                model_reply = ollama.chat(
                    model="llama3.2",
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": complete_prompt}
                    ],
                    options={"temperature": 0.1}
                )
                
                ai_answer_string = model_reply['message']['content']
                st.markdown(ai_answer_string)
                
                # Neatly render multiple sources inside structured metadata expansion bubbles
                with st.expander("📍 View Verification Provenance Chunks"):
                    for text, source_id in zip(retrieved_chunks_list, source_reference_keys):
                        st.markdown(f"**Source Target Reference ID:** `{source_id}`")
                        st.caption(f"Text Segment Match: *\"{text}\"*")
                        st.markdown("---")
                
        st.session_state.messages.append({"role": "assistant", "content": ai_answer_string})
