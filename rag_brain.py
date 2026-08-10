import os
import chromadb
from chromadb.utils import embedding_functions
import ollama

def generate_grounded_answer(user_question):
    # 1. Open our existing local vector database folder
    chroma_client = chromadb.PersistentClient(path="./my_vector_db")
    default_ef = embedding_functions.DefaultEmbeddingFunction()
    
    collection = chroma_client.get_collection(
        name="company_rules", 
        embedding_function=default_ef
    )

    # 2. Search the database for the closest match to the question
    results = collection.query(
        query_texts=[user_question],
        n_results=1
    )

    # Extract the text fragment and its record ID
    matched_text = results['documents']
    matched_id = results['ids']

    # 3. Construct the exact instructions for your local model
    system_instruction = (
        "You are a strict company support bot. Answer the user's question using ONLY the provided Context. "
        "Do not invent facts. You MUST end your sentence by adding the exact Source ID wrapped in square brackets, "
        f"like this: {matched_id}."
    )
    
    ai_prompt = f"Context: {matched_text}\n\nQuestion: {user_question}"

    print("🤖 Sending context and question to local Ollama model...")

    # 4. Call your completely free local AI brain
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": ai_prompt}
        ],
        options={"temperature": 0.0} # Keep it completely factual
    )

    # 5. Extract the final text result from the local model
    final_answer = response['message']['content']
    return final_answer

if __name__ == "__main__":
    # Test our local RAG system with a fresh question!
    test_question = "Can remote workers get money for a desk or chair?"
    print(f"\n❓ User Question: '{test_question}'")
    
    answer = generate_grounded_answer(test_question)
    print(f"\n✨ Grounded Answer from Local AI:\n{answer}")
