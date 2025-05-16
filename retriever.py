import os
import glob
import chromadb
from chromadb.utils import embedding_functions

# Default embedding model if not specified in environment
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")


def get_latest_package_name():
    chroma_dirs = sorted(glob.glob(".chroma_*/"), key=os.path.getmtime, reverse=True)
    if not chroma_dirs:
        raise RuntimeError("No indexed package found. Please ingest a package first.")
    return chroma_dirs[0].replace(".chroma_", "").strip("/\\")


def retrieve_relevant_chunks(query: str, package_name: str = None, k: int = 5):
    if package_name is None:
        package_name = get_latest_package_name()
    db_dir = f".chroma_{package_name}"
    if not os.path.exists(db_dir):
        raise RuntimeError(
            f"No index found for package '{package_name}'. Please ingest it first."
        )
    client = chromadb.PersistentClient(path=db_dir)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name=OPENAI_EMBEDDING_MODEL,
    )
    collection = client.get_or_create_collection(
        name="docs",
        embedding_function=openai_ef,
    )
    results = collection.query(query_texts=[query], n_results=k)
    return results["documents"][0] if results["documents"] else []
