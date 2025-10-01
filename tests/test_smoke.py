from muras import evaluate, ImageTextRelevance
from muras.metrics import Sample

def test_smoke():
    s = Sample(query="What is shown?", contexts_image_paths=["/tmp/fake.png"], answer="foo")
    out = evaluate([s])
    assert "metrics" in out and "image_text_relevance" in out["metrics"]
