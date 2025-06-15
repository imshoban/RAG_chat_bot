import chromadb
from chromadb.utils import embedding_functions


# Initialize ChromaDB client with persistence
client = chromadb.PersistentClient(path="chroma_db")

# Configure sentence transformer embeddings
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create or get existing collection
collection = client.get_or_create_collection(
    name="vector_db",
    embedding_function=sentence_transformer_ef
)


