# muras
Base repo for muras eval multimodal-RAG suite

cd into muras folder
python -m build

This creates a dist/ folder with:
muras-0.0.1-py3-none-any.whl (wheel - installable binary)
muras-0.0.1.tar.gz (source distribution)

pip install -e /workspace/muras


# 1. Install in editable mode (do this once)
pip install -e .

# 2. Make your code changes
# (edit files in src/muras/)

# 3. Test immediately (no reinstall needed!)
pytest /workspace/muras/tests/

# 4. When ready to publish, build it
python -m build