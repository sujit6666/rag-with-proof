# 🔒 Helix Premium RAG Core

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://helix-rag-engine.streamlit.app/)

An enterprise-grade, privacy-first Document Intelligence System built to ingest PDF files, index them into a high-performance vector database, and return answers with strict, undeniable citation proof keys.

🔗 **Live App:** https://helix-rag-engine.streamlit.app/

## 🚀 Key Architectural Features
- **Semantic Vector Storage:** Fractures unstructured document text layouts into isolated sliding contextual chunks using an in-memory ChromaDB search index.
- **Dynamic Context Window Expansion:** Dynamically retrieves and stitches the top 4 closest matching paragraph chunks to eliminate LLM hallucinations.
- **Adaptive Visual UI Syncing:** Injects custom theme-adaptive CSS overrides to maintain high-contrast, bold readability in both Light and Dark modes.
- **Strict Verification Provenance:** Forces the model to append precise data reference tracking keys and source snippets to every generated response bubble.
- **Hybrid Cloud Fallback:** Prioritizes local Ollama inference while seamlessly routing requests through Groq cloud execution for live web deployments.

## 🛠️ Project File Components
- `app.py`: The master document ingestion and conversational interface dashboard.
- `requirements.txt`: Production dependency specifications for cloud hosting environments.
- `rag_brain.py`: The core backend contextual search query matching execution script.
- `company_policy.txt`: The sample policy dataset file used for pipeline architecture testing.
- `.gitignore`: The structural filtering matrix protecting local virtual environments and caches.

## ⚙️ Quick Installation Guide
To execute this private retrieval framework pipeline locally on your computer device terminal:

```cmd
# Activate your python sandbox environment
venv\Scripts\activate

# Install the necessary parsing and modeling dependencies
pip install -r requirements.txt

# Launch the interactive web browser dashboard canvas
streamlit run app.py