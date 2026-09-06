"""
Step 3: Create Embeddings and Index
"""
import os
import json
import ollama
import chromadb

PAGES_FILE = "data/cache/all_pages.json"
CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "ikea_instructions"


def get_embedding(text):
    """Get embedding from Local Ollama model."""
    response = ollama.embeddings(
        model="nomic-embed-text:v1.5",
        prompt=text
    )
    return response["embedding"]


def main():
    # Load page descriptions
    with open(PAGES_FILE, "r") as f:
        pages = json.load(f)
    print(f"Loaded {len(pages)} pages")
    
    # Initialize ChromaDB with persistent storage
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    # Delete existing collection if present
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
    except:
        pass
    
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"Created collection: {COLLECTION_NAME}")
    
    # Index each page
    print("\nCreating embeddings...")
    for i, page in enumerate(pages):
        # Combine source info with description
        text_to_embed = f"Source: {page['source_pdf']}, Page {page['page_number']}\n\n{page['description']}"
        
        embedding = get_embedding(text_to_embed)
        
        collection.add(
            ids=[f"page_{i}"],
            embeddings=[embedding],
            documents=[page["description"]],
            metadatas=[{
                "source_pdf": page["source_pdf"],
                "page_number": page["page_number"],
                "image_path": page["image_path"]
            }]
        )
        
        if (i + 1) % 5 == 0 or i == len(pages) - 1:
            print(f"  Indexed {i + 1}/{len(pages)} pages")
    
    print(f"\n✓ Index created with {collection.count()} documents")
    
    # Quick retrieval test
    print("\nTesting retrieval...")
    test_query = "How do I attach the legs?"
    query_embedding = get_embedding(test_query)
    results = collection.query(query_embeddings=[query_embedding], n_results=2)
    
    print(f"Query: '{test_query}'")
    for meta in results["metadatas"][0]:
        print(f"  → {meta['source_pdf']}, Page {meta['page_number']}")


if __name__ == "__main__":
    main()