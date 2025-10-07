"""
Example: Using vector stores for multimodal retrieval

This demonstrates how to use FAISS and Chroma vector stores to:
1. Index text and image documents
2. Search using text queries
3. Search using image queries
4. Perform cross-modal retrieval
"""

from muras import CLIPEmbedder, FAISSVectorStore, ChromaVectorStore, Document

# Create some sample documents
documents = [
    Document(
        id="doc1",
        text="Horse racing is a popular equestrian sport",
        metadata={"category": "sports", "source": "encyclopedia"}
    ),
    Document(
        id="doc2",
        text="Swimming is a great form of exercise",
        metadata={"category": "sports", "source": "encyclopedia"}
    ),
    Document(
        id="doc3",
        text="Cars are vehicles used for transportation",
        metadata={"category": "transportation", "source": "encyclopedia"}
    ),
    Document(
        id="doc4",
        image_path="../sample/horse_racing.jpg",
        metadata={"category": "sports", "source": "image"}
    ),
    Document(
        id="doc5",
        image_path="../sample/swimming_man.jpg",
        metadata={"category": "sports", "source": "image"}
    ),
]

print("=" * 60)
print("VECTOR STORE EXAMPLES")
print("=" * 60)

# Initialize embedder
print("\nInitializing CLIP embedder...")
embedder = CLIPEmbedder()

# ============================================================================
# Example 1: FAISS Vector Store
# ============================================================================
print("\n" + "=" * 60)
print("EXAMPLE 1: FAISS Vector Store")
print("=" * 60)

vector_store = FAISSVectorStore(
    embedder=embedder,
    persist_directory="./faiss_store"
)

print(f"\nAdding {len(documents)} documents to FAISS...")
added_ids = vector_store.add_documents(documents)
print(f"   Added documents: {added_ids}")
print(f"   Total documents in store: {vector_store.get_document_count()}")

# Text query search
query = "people racing on horses"
print(f"\nSearching for: '{query}'")
results = vector_store.search_by_text(query, top_k_text=1, top_k_image=1)

for i, result in enumerate(results, 1):
    print(f"\n   Result {i}:")
    print(f"   - Document ID: {result.document.id}")
    print(f"   - Similarity Score: {result.score:.4f}")
    if result.document.text:
        print(f"   - Text: {result.document.text[:60]}...")
    if result.document.image_path:
        print(f"   - Image: {result.document.image_path}")
    print(f"   - Metadata: {result.document.metadata}")

# Image query search
image_query = "../sample/horse_racing2.jpg"
print(f"\nSearching with image: '{image_query}'")
results = vector_store.search_by_image(image_query, top_k_text=1, top_k_image=1)

for i, result in enumerate(results, 1):
    print(f"\n   Result {i}:")
    print(f"   - Document ID: {result.document.id}")
    print(f"   - Similarity Score: {result.score:.4f}")
    if result.document.text:
        print(f"   - Text: {result.document.text[:60]}...")
    if result.document.image_path:
        print(f"   - Image: {result.document.image_path}")

# Save to disk
print("\nSaving FAISS vector store...")
vector_store.save()
print("   ✅ Saved to ./faiss_store")

# Load from disk
print("\nLoading FAISS vector store from disk...")
loaded_store = FAISSVectorStore.load(
    embedder=embedder,
    path="./faiss_store"
)
print(f"   ✅ Loaded {loaded_store.get_document_count()} documents")

# ============================================================================
# Example 2: Chroma Vector Store
# ============================================================================
print("\n" + "=" * 60)
print("EXAMPLE 2: Chroma Vector Store")
print("=" * 60)

try:
    vector_store = ChromaVectorStore(
        embedder=embedder,
        persist_directory="./chroma_db",
        collection_name="demo"
    )
    
    print(f"\nAdding {len(documents)} documents to Chroma...")
    added_ids = vector_store.add_documents(documents)
    print(f"   Added documents: {added_ids}")
    print(f"   Total documents in store: {vector_store.get_document_count()}")
    
    # Search with metadata filter
    query = "racing"
    print(f"\nSearching for: '{query}' (with metadata filter: category='sports')")
    results = vector_store.search_by_text(
        query, 
        top_k_text=3, 
        top_k_image=3, 
        filter_metadata={"category": "sports"}
    )
    
    for i, result in enumerate(results, 1):
        print(f"\n   Result {i}:")
        print(f"   - Document ID: {result.document.id}")
        print(f"   - Similarity Score: {result.score:.4f}")
        print(f"   - Category: {result.document.metadata.get('category')}")
        if result.document.text:
            print(f"   - Text: {result.document.text[:50]}...")
        if result.document.image_path:
            print(f"   - Image: {result.document.image_path}")
    
    # Data is automatically persisted
    print("\nData automatically persisted to ./chroma_db")
    
    # Load from disk
    print("\nLoading Chroma vector store from disk...")
    loaded_chroma = ChromaVectorStore.load(
        embedder=embedder,
        persist_directory="./chroma_db",
        collection_name="demo"
    )
    print(f"   ✅ Loaded {loaded_chroma.get_document_count()} documents")

except Exception as e:
    print(f"❌ Chroma example failed: {e}")
    print("   Make sure to install: pip install chromadb")


print("\n✅ Vector store examples complete!")
