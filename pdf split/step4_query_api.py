"""
Step 4: Query the Multimodal RAG System
"""
import os
import sys
import base64
from openai import OpenAI
import chromadb

client = OpenAI()

CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "ikea_instructions"
TOP_K = 3

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)


def get_embedding(text):
    """Get embedding from OpenAI."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


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
    """Generate answer using GPT-4o with retrieved images."""
    if not retrieved_results["ids"][0]:
        return "No relevant pages found.", []
    
    # Build image content for GPT-4o
    image_content = []
    sources = []
    
    for doc, metadata in zip(retrieved_results["documents"][0], retrieved_results["metadatas"][0]):
        image_path = metadata["image_path"]
        
        if not os.path.exists(image_path):
            continue
        
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")
        
        image_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
        })
        image_content.append({
            "type": "text",
            "text": f"[Page {metadata['page_number']} from {metadata['source_pdf']}]"
        })
        
        sources.append(f"{metadata['source_pdf']}, Page {metadata['page_number']}")
    
    # Generate with GPT-4o
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """You answer questions about IKEA assembly based on instruction pages.
Be specific, reference page numbers when helpful, and say if something isn't clear from the images."""
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Question: {query}\n\nRelevant instruction pages:"},
                    *image_content,
                    {"type": "text", "text": "Answer based on these pages."}
                ]
            }
        ],
        max_tokens=600
    )
    
    return response.choices[0].message.content, sources


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
        print("ERROR: Index not found. Run step2_index.py first.")
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