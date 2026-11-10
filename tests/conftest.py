import os

for var, valor in {
    "GEMINI_API_KEY": "test",
    "GEMINI_MODEL": "test",
    "PINECONE_API_KEY": "test",
    "PINECONE_INDEX": "test",
    "EMBEDDING_MODEL": "test",
}.items():
    os.environ.setdefault(var, valor)
