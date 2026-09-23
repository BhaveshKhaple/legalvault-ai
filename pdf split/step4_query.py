"""
Step 4: Query the Multimodal RAG System (100% Local with Ollama)
"""
import os
import sys
import ollama
import chromadb

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
    """Generate answer using local Gemma model with retrieved images."""
    if not retrieved_results["ids"][0]:
        return "No relevant pages found.", []
    
    # We will build a prompt with all retrieved pages.
    image_paths = []
    sources = []
    
    # Create text context from metadata
    context_text = f"Question: {query}\n\nRelevant instruction pages:\n"
    
    for doc, metadata in zip(retrieved_results["documents"][0], retrieved_results["metadatas"][0]):
        image_path = metadata["image_path"]
        
        if not os.path.exists(image_path):
            continue
            
        image_paths.append(image_path)
        sources.append(f"{metadata['source_pdf']}, Page {metadata['page_number']}")
        context_text += f"[Page {metadata['page_number']} from {metadata['source_pdf']}]\n"

    context_text += "\nAnswer based on these pages."
    
    response = ollama.chat(
        model="gemma4:e2b-it-qat",
        messages=[
            {
                "role": "system",
                # Include the reasoning token if your model version supports it
                "content": """<|think|>
You answer questions about IKEA assembly based on instruction pages.
Be specific, reference page numbers when helpful, and say if something isn't clear from the images."""
            },
            {
                "role": "user",
                "content": context_text,
                "images": image_paths # Ollama handles the file paths directly
            }
        ]
    )
    
    return response['message']['content'], sources


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