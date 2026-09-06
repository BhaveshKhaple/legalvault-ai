"""
Step 4: Query the Multimodal RAG System (Gemini + Ollama)
"""
import os
import sys
import ollama
import chromadb
from google import genai
from google.genai import types

# Initialize the Gemini client (Requires GEMINI_API_KEY environment variable)
client = genai.Client()

CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "ikea_instructions"
TOP_K = 3

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)


def get_embedding(text):
    """Get embedding from Local Ollama model to match Step 3."""
    response = ollama.embeddings(
        model="nomic-embed-text:v1.5",
        prompt=text
    )
    return response["embedding"]


def retrieve(query, top_k=TOP_K):
    """Find relevant instruction pages."""
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    query_embedding = get_embedding(query)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas"]
    )
    return results


def generate_answer(query, retrieved_results):
    """Generate answer using Gemini with retrieved images."""
    if not retrieved_results["ids"][0]:
        return "No relevant pages found.", []
    
    # Build content payload for Gemini
    contents = [f"Question: {query}\n\nRelevant instruction pages:\n"]
    sources = []
    
    for doc, metadata in zip(retrieved_results["documents"][0], retrieved_results["metadatas"][0]):
        image_path = metadata["image_path"]
        
        if not os.path.exists(image_path):
            continue
        
        # Read the raw bytes of the image directly
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        # Add the image and its text identifier to the prompt
        contents.append(types.Part.from_bytes(data=image_bytes, mime_type="image/png"))
        contents.append(f"[Page {metadata['page_number']} from {metadata['source_pdf']}]")
        
        sources.append(f"{metadata['source_pdf']}, Page {metadata['page_number']}")
    
    contents.append("Answer based on these pages.")
    
    # Generate response with Gemini 2.5 Flash
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction="""You answer questions about IKEA assembly based on instruction pages.
Be specific, reference page numbers when helpful, and say if something isn't clear from the images.""",
            max_output_tokens=600
        )
    )
    
    return response.text, sources


def answer_question(query):
    """Full RAG pipeline."""
    print(f"\nQuery: {query}")
    print("-" * 50)
    
    print("Retrieving relevant pages...")
    results = retrieve(query)
    print(f"  Found {len(results['ids'][0])} pages")
    
    print("Generating answer...")
    answer, sources = generate_answer(query, results)
    
    return answer, sources


def main():
    # Check index exists
    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        print(f"Connected to index ({collection.count()} documents)")
    except:
        print("ERROR: Index not found. Run step3_index.py first.")
        return
    
    # Single query from command line
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        answer, sources = answer_question(query)
        print(f"\nAnswer:\n{answer}")
        print(f"\nSources: {', '.join(sources)}")
        return
    
    # Interactive mode
    print("\nAsk questions about IKEA assembly. Type 'quit' to exit.\n")
    
    while True:
        try:
            query = input("Question: ").strip()
        except KeyboardInterrupt:
            break
        
        if not query or query.lower() in ["quit", "exit"]:
            break
        
        answer, sources = answer_question(query)
        print(f"\nAnswer:\n{answer}")
        print(f"\nSources: {', '.join(sources)}\n")


if __name__ == "__main__":
    main()