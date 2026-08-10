# 🔒 Helix Premium RAG Core

An enterprise-grade, privacy-first Document Intelligence System built to ingest local PDF files, index them into a mathematical vector database, and return answers with strict, undeniable citation proof keys completely offline.

## 🚀 Key Architectural Features
- **Semantic Vector Storage:** Fractures unstructured document text layouts into isolated sliding contextual chunks using an offline mathematical search index.
- **Dynamic Context Window Expansion:** Dynamically retrieves and stitches the top 4 closest matching paragraph chunks to eliminate LLM hallucinations.
- **Adaptive Visual UI Syncing:** Injects custom theme-adaptive CSS overrides to maintain high-contrast, bold readability in both Light and Dark modes.
- **Strict Verification Provenance:** Forces the local model to append precise data reference tracking keys to every generated response bubble.

## 🛠️ Project File Components
- `app.py`: The master multi-file uploading chat interface web app dashboard.
- `rag_brain.py`: The core backend contextual search query matching execution script.
- `company_policy.txt`: The initial sample rule file used for pipeline architecture testing.
- `.gitignore`: The structural filtering matrix layout file protecting heavy internal system dependencies.

## ⚙️ Quick Installation Guide
To execute this private retrieval framework pipeline locally on your computer device terminal:

```cmd
# Activate your python sandbox environment
venv\Scripts\activate

# Install the necessary parsing and modeling dependencies
pip install chromadb ollama streamlit pypdf

# Launch the interactive web browser dashboard canvas
streamlit run app.py
```
