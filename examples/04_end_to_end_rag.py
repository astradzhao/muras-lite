"""
Example: End-to-end Multimodal RAG Pipeline

This demonstrates a complete multimodal RAG workflow:
1. Embed documents (text + images)
2. Store in vector database
3. Retrieve relevant context
4. Generate answer with VLM
"""

from muras import (
    CLIPEmbedder,
    FAISSVectorStore,
    Document,
    Qwen2_5VLGenerator,
    GenerationInput
)

print("=" * 60)
print("END-TO-END MULTIMODAL RAG")
print("=" * 60)

# Step 1: Initialize components (we use CLIP with FAISS as example)
print("\nStep 1: Initializing components...")
embedder = CLIPEmbedder()
vector_store = FAISSVectorStore(embedder=embedder, persist_directory="./rag_store")

# For generation, we provide an example for Qwen2_5VLGenerator
# - Qwen2_5VLGenerator (best multimodal performance)

print("   Loading VLM generator (this may take a moment)...")
try:
    generator = Qwen2_5VLGenerator(
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        # load_in_4bit=True - optional for quantization
        # load_in_8bit=True - optional for quantization
    )
    print("   Generator ready!")
except Exception as e:
    print(f"   Could not load generator: {e}")
    print("   Install with: pip install transformers accelerate bitsandbytes")
    generator = None

# Step 2: Index documents
print("\nStep 2: Indexing documents...")
documents = [
    Document(
        id="doc1",
        text="Horse racing is a popular equestrian sport where horses and jockeys compete. In 2025, there was a massive horse race in Japan.",
        metadata={"category": "sports", "topic": "horses"}
    ),
    Document(
        id="doc2",
        text="Swimming is an excellent cardiovascular exercise that works all muscle groups.",
        metadata={"category": "sports", "topic": "swimming"}
    ),
    Document(
        id="doc3",
        text="The horse is a magnificent animal known for its speed and grace.",
        metadata={"category": "animals", "topic": "horses"}
    ),
    Document(
        id="doc4",
        image_path="../sample/horse_racing.jpg",
        metadata={"category": "sports", "topic": "horses", "type": "image"}
    ),
    Document(
        id="doc5",
        image_path="../sample/swimming_man.jpg",
        metadata={"category": "sports", "topic": "swimming", "type": "image"}
    ),
]

vector_store.add_documents(documents)
print(f"   ✓ Indexed {len(documents)} documents")

# Step 3: User query
print("\nStep 3: User query...")
query = "Where in the world was the massive horse race held in 2025?"
print(f"   Query: '{query}'")

# Step 4: Retrieve relevant context
print("\nStep 4: Retrieving relevant context...")
results = vector_store.search_by_text(
    query,
    top_k_text=2,
    top_k_image=2
)

print(f"   Found {len(results)} relevant items:")
context_texts = []
context_images = []

for i, result in enumerate(results, 1):
    print(f"\n   {i}. {result.document.id} (score: {result.score:.3f})")
    if result.document.text:
        print(f"      Text: {result.document.text[:60]}...")
        context_texts.append(result.document.text)
    if result.document.image_path:
        print(f"      Image: {result.document.image_path}")
        context_images.append(result.document.image_path)

# Step 5: Generate answer
if generator:
    print("\nStep 5: Generating answer with VLM...")
    
    generation_input = GenerationInput(
        query=query,
        context_texts=context_texts,
        context_image_paths=context_images,
        max_tokens=256,
        temperature=0.7,
        system_prompt=(
            "You are a helpful assistant. Answer the question based on the "
            "provided text and image context. Be informative and accurate."
        )
    )
    
    output = generator.generate(generation_input)
    
    print("\n" + "=" * 60)
    print("GENERATED ANSWER")
    print("=" * 60)
    print(f"\n{output.generated_text}\n")
    print("=" * 60)
    
    print(f"\nMetadata:")
    print(f"  Model: {output.model_name}")
    print(f"  Tokens: {output.tokens_used}")
    print(f"  Context: {output.metadata['num_context_texts']} texts, {output.metadata['num_context_images']} images")
else:
    print("\nStep 5: Skipping generation (generator not loaded)")
    print("   Install dependencies to enable generation:")
    print("   pip install transformers accelerate bitsandbytes")

# Step 6: Save vector store (optional)
print("\nStep 6: Saving vector store...")
vector_store.save()
print("   ✓ Saved to ./rag_store")

print("\nEnd-to-end RAG pipeline complete!")



