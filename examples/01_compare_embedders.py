"""
Example: Compare different embedders on the same data

This shows how to use different embedders (CLIP, BLIP, Sentence-Transformers)
to evaluate the same samples and compare their performance.
"""

from muras import Sample, evaluate, CLIPEmbedder, SigLIPEmbedder, SentenceTransformerEmbedder, SigLIP2Embedder, BLIP2Embedder, ColPaliEmbedder

samples = [
    Sample(
        query="The people are racing on horses",
        contexts_text=[
            "Horse racing is a popular sport",
            "The man is swimming",
            "Dont sleep in the car please"
        ],
        contexts_image_paths=[
            "../sample/horse_racing.jpg",
            "../sample/swimming_man.jpg",
            "../sample/horse_racing2.jpg"
        ],
        answer="Horses are racing"
    ),
    Sample(
        query="The people are racing on horses",
        contexts_text=[
            "Horse racing is a popular sport",
            "The man is swimming",
            "hello world"
        ],
        contexts_image_paths=[
            "../sample/horse_racing.jpg",
            "../sample/swimming_man.jpg",
        ],
        answer="Horses are racing"
    ),
    Sample(
        query="The people are racing on horses",
        contexts_text=[
            "Horse racing is a popular sport",
            "The man is swimming",
            "Don't sleep in the car please"
        ],
        contexts_image_paths=[
            "../sample/horse_racing.jpg",
            "../sample/swimming_man.jpg",
        ],
        answer="Horses are racing"
    ),
    Sample(
        query="The people are racing on horses",
        contexts_text=[
            "Horse racing is a popular sport",
            "The man is swimming",
            "all horses are beautiful"
        ],
        contexts_image_paths=[
            "../sample/horse_racing.jpg",
            "../sample/swimming_man.jpg",
        ],
        answer="Horses are racing"
    ),
    Sample(
        query="The people are racing on horses",
        contexts_text=[
            "Horse racing is a popular sport",
            "The man is swimming",
            "hello world1"
        ],
        contexts_image_paths=[
            "../sample/horse_racing.jpg",
            "../sample/swimming_man.jpg",
        ],
        answer="Horses are racing"
    ),
]

print("=" * 60)
print("COMPARING DIFFERENT EMBEDDERS")
print("=" * 60)

# Test different embedders
embedders = {
    # "CLIP": CLIPEmbedder(),
    # "BLIP2": BLIP2Embedder(),
    # "SigLIP": SigLIPEmbedder(),
    # "SigLIP2": SigLIP2Embedder(),
    # "Sentence-Transformers": SentenceTransformerEmbedder(),
    "ColPali": ColPaliEmbedder()
}

results = {}

for name, embedder in embedders.items():
    print(f"\n🔍 Testing {name}...")
    try:
        result = evaluate(samples, embedder=embedder)
        results[name] = result
        
        print(f"Text Relevance: {result['aggregate']['text_relevance_avg']:.3f}")
        print(f"Image Relevance: {result['aggregate']['image_relevance_avg']:.3f}")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        results[name] = None


# for sample in results["SigLIP2"]["samples"]:
#     print(sample)
#     print("-" * 60)

#print(results["CLIP"]["samples"])

