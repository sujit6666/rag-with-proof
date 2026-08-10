import os
import chromadb
from chromadb.utils import embedding_functions

def build_knowledge_base():
    # 1. Initialize a persistent database folder on your local machine
    # This creates a folder named 'my_vector_db' to store your data safely.
    chroma_client = chromadb.PersistentClient(path="./my_vector_db")

    # 2. Setup the mathematical framework that converts text into numeric concepts
    # We use a built-in lightweight model that runs completely offline on your computer.
    default_ef = embedding_functions.DefaultEmbeddingFunction()

    # 3. Create or access a specific storage bucket within our database
    collection = chroma_client.get_or_create_collection(
        name="company_rules", 
        embedding_function=default_ef
    )

    # 4. Check if our text policy document exists before opening it
    if not os.path.exists("company_policy.txt"):
        print("❌ Error: company_policy.txt was not found in this folder!")
        return

    # 5. Open and read your private policy text document line by line
    with open("company_policy.txt", "r", encoding="utf-8") as file:
        lines = file.readlines()

    # 6. Clean up trailing white spaces and ignore any completely blank lines
    documents_to_add = [line.strip() for line in lines if line.strip()]
    
    # 7. Generate a distinct tracking ID for each rule (e.g., 'rule_0', 'rule_1')
    doc_ids = [f"rule_{i}" for i in range(len(documents_to_add))]

    print("🔄 Indexing files into the local vector database...")
    
    # 8. Store the raw text documents directly inside our database bucket
    collection.add(
        documents=documents_to_add,
        ids=doc_ids
    )
    print("✅ Successfully saved and indexed all corporate policies!")

    # 9. Test the engine immediately using a sample user inquiry
    user_query = "What is the daily food allowance limit?"
    print(f"\n🔍 Testing Database Query: '{user_query}'")
    
    results = collection.query(
        query_texts=[user_query],
        n_results=1  # Instruct the engine to bring back only the single closest match
    )

    # 10. Extract and print the exact document text found by the database algorithm
    matched_text = results['documents'][0][0]
    matched_id = results['ids'][0][0]
    print(f"📦 Best Match Found: {matched_text} (ID: {matched_id})")

if __name__ == "__main__":
    build_knowledge_base()
