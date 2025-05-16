import os
import tempfile
import requests
import tarfile
import zipfile
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
import subprocess
import re

# Default embedding model if not specified in environment
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")


def download_and_extract_package(package_name_or_repo_url, dest_dir):
    # Check if the input is a GitHub URL
    if re.match(r"https?://github\.com/[\w-]+/[\w-]+", package_name_or_repo_url):
        # Clone the repository
        subprocess.run(["git", "clone", package_name_or_repo_url, dest_dir], check=True)
    else:
        # Treat as a PyPI package
        resp = requests.get(f"https://pypi.org/pypi/{package_name_or_repo_url}/json")
        resp.raise_for_status()
        data = resp.json()
        url = data["urls"][0]["url"]
        filename = url.split("/")[-1]
        pkg_path = os.path.join(dest_dir, filename)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(pkg_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        # Extract
        if filename.endswith(".zip"):
            with zipfile.ZipFile(pkg_path, "r") as zip_ref:
                zip_ref.extractall(dest_dir)
        elif filename.endswith((".tar.gz", ".tgz")):
            with tarfile.open(pkg_path, "r:gz") as tar_ref:
                tar_ref.extractall(dest_dir)
        else:
            raise ValueError("Unknown archive format: " + filename)


def collect_text_files(root_dir):
    # Collect .py, .md, .rst, .txt files
    files = []
    for ext in (".py", ".md", ".rst", ".txt"):
        files.extend(Path(root_dir).rglob(f"*{ext}"))
    return files


def chunk_text(text, max_length=1000):
    # Simple chunking by lines
    lines = text.splitlines()
    chunks = []
    chunk = []
    length = 0
    for line in lines:
        if length + len(line) > max_length and chunk:
            chunks.append("\n".join(chunk))
            chunk = []
            length = 0
        chunk.append(line)
        length += len(line)
    if chunk:
        chunks.append("\n".join(chunk))
    return chunks


def ingest_and_index_package(package_name_or_repo_url: str):
    # Use the last part of the URL or package name as the identifier
    identifier = package_name_or_repo_url.split("/")[-1].replace(".git", "")
    db_dir = f".chroma_{identifier}"
    if os.path.exists(db_dir):
        return  # Already indexed
    with tempfile.TemporaryDirectory() as tmpdir:
        download_and_extract_package(package_name_or_repo_url, tmpdir)
        # Find the main extracted folder
        subdirs = [d for d in Path(tmpdir).iterdir() if d.is_dir()]
        if not subdirs:
            raise RuntimeError("No extracted folder found.")
        root = subdirs[0]
        files = collect_text_files(root)
        docs = []
        metadatas = []
        for file in files:
            try:
                text = file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for chunk in chunk_text(text):
                docs.append(chunk)
                metadatas.append({"source": str(file)})

        if not docs:
            raise RuntimeError(
                f"No valid documents found in {package_name_or_repo_url}. Please check if the package/repository contains any Python files, markdown files, or documentation."
            )

        # Index with ChromaDB
        client = chromadb.PersistentClient(path=db_dir)
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=OPENAI_EMBEDDING_MODEL,
        )
        collection = client.get_or_create_collection(
            name="docs",
            embedding_function=openai_ef,
        )
        collection.add(
            documents=docs, metadatas=metadatas, ids=[str(i) for i in range(len(docs))]
        )
