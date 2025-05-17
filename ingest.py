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
from typing import List, Set, Dict
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)

# Initialize rich console
console = Console()

# Default embedding model if not specified in environment
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")

# Python-related file extensions to process
PYTHON_EXTENSIONS = {
    # Python source files
    ".py": "python",
    # Documentation
    ".md": "markdown",
    ".rst": "restructuredtext",
    ".txt": "text",
    # Python project configuration
    ".pyi": "python-stub",
    "py.typed": "python-typed",
    "pyproject.toml": "python-config",
    "setup.py": "python-config",
    "setup.cfg": "python-config",
    "requirements.txt": "python-deps",
    "requirements-dev.txt": "python-deps",
    "requirements-test.txt": "python-deps",
    "MANIFEST.in": "python-config",
    # Type hints
    ".pyi": "python-stub",
    # Documentation
    "README.md": "documentation",
    "README.rst": "documentation",
    "CHANGELOG.md": "documentation",
    "CHANGELOG.rst": "documentation",
    "LICENSE": "documentation",
    "AUTHORS": "documentation",
    "CONTRIBUTING.md": "documentation",
    "CONTRIBUTING.rst": "documentation",
}

# Directories to ignore
IGNORE_DIRS = {
    # Build and cache directories
    ".git",
    "__pycache__",
    "venv",
    ".venv",
    "env",
    ".env",
    "build",
    "dist",
    "target",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".coverage",
    "htmlcov",
    "site-packages",
    "node_modules",  # Sometimes present in Python projects
    ".tox",
    ".eggs",
    "*.egg-info",
    "*.egg",
    # Documentation and example directories
    "docs",
    "doc",
    "documentation",
    "examples",
    "example",
}

# Patterns to ignore in directory names
IGNORE_PATTERNS = {
    "*docs*",
    "*doc*",
    "*documentation*",
    "*example*",
    "*examples*",
}


def download_and_extract_package(package_name_or_repo_url, dest_dir):
    # Check if the input is a GitHub URL
    if re.match(r"https?://github\.com/[\w-]+/[\w-]+", package_name_or_repo_url):
        # Clone the repository
        console.print(f"[bold blue]Cloning repository:[/] {package_name_or_repo_url}")
        subprocess.run(["git", "clone", package_name_or_repo_url, dest_dir], check=True)
        return dest_dir  # Return the directory where the repo was cloned
    else:
        # Treat as a PyPI package
        console.print(f"[bold blue]Downloading package:[/] {package_name_or_repo_url}")
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
        console.print(f"[bold blue]Extracting package...[/]")
        if filename.endswith(".zip"):
            with zipfile.ZipFile(pkg_path, "r") as zip_ref:
                zip_ref.extractall(dest_dir)
        elif filename.endswith((".tar.gz", ".tgz")):
            with tarfile.open(pkg_path, "r:gz") as tar_ref:
                tar_ref.extractall(dest_dir)
        else:
            raise ValueError("Unknown archive format: " + filename)

        # Find the main package directory
        # For PyPI packages, it's usually the directory with the same name as the package
        package_name = package_name_or_repo_url.lower()
        package_dir = None

        # First try to find a directory matching the package name
        for item in Path(dest_dir).iterdir():
            if item.is_dir() and item.name.lower() == package_name:
                package_dir = item
                break

        # If not found, use the first directory that contains Python files
        if not package_dir:
            for item in Path(dest_dir).iterdir():
                if item.is_dir() and any(item.rglob("*.py")):
                    package_dir = item
                    break

        # If still not found, use the root directory
        if not package_dir:
            package_dir = Path(dest_dir)

        return str(package_dir)


def should_process_file(file_path: Path) -> bool:
    """Check if a file should be processed based on its path and extension."""
    # Check if file is in an ignored directory
    for part in file_path.parts:
        # Check exact directory names
        if part in IGNORE_DIRS:
            return False

        # Check directory name patterns
        if any(
            part.lower().startswith(pattern.replace("*", ""))
            for pattern in IGNORE_PATTERNS
        ):
            return False

    # Check if file has a supported extension or is a special Python file
    return (
        file_path.suffix.lower() in PYTHON_EXTENSIONS
        or file_path.name in PYTHON_EXTENSIONS
    )


def collect_text_files(root_dir: str) -> List[Dict]:
    """Collect all Python-related files recursively from the root directory."""
    files = []
    root_path = Path(root_dir)

    # Count total files first
    total_files = sum(1 for _ in root_path.rglob("*") if _.is_file())
    processed_files = 0
    skipped_files = 0

    console.print(f"\n[bold blue]Scanning for Python files...[/]")

    # Walk through all files recursively
    for file_path in root_path.rglob("*"):
        if file_path.is_file():
            processed_files += 1
            if should_process_file(file_path):
                try:
                    # Try to read the file with different encodings
                    for encoding in ["utf-8", "latin-1", "cp1252"]:
                        try:
                            text = file_path.read_text(encoding=encoding)
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        # If all encodings fail, skip the file
                        skipped_files += 1
                        continue

                    # Get file type from extension or name
                    file_type = PYTHON_EXTENSIONS.get(
                        file_path.suffix.lower(),
                        PYTHON_EXTENSIONS.get(file_path.name, "unknown"),
                    )

                    # Add file info
                    files.append(
                        {
                            "path": str(file_path),
                            "content": text,
                            "type": file_type,
                            "relative_path": str(file_path.relative_to(root_path)),
                        }
                    )
                except Exception as e:
                    console.print(
                        f"[yellow]Warning:[/] Error processing {file_path}: {e}"
                    )
                    skipped_files += 1
                    continue
            else:
                skipped_files += 1

            # Print progress every 100 files
            if processed_files % 100 == 0:
                console.print(f"Processed {processed_files}/{total_files} files...")

    # Print final statistics
    console.print(f"\n[bold green]File processing complete![/]")
    console.print(f"Total files scanned: {total_files}")
    console.print(f"Files processed: {len(files)}")
    console.print(f"Files skipped: {skipped_files}")

    # Print file type statistics
    file_types = {}
    for file in files:
        file_types[file["type"]] = file_types.get(file["type"], 0) + 1

    console.print("\n[bold blue]File type statistics:[/]")
    for file_type, count in sorted(file_types.items()):
        console.print(f"{file_type}: {count} files")

    return files


def chunk_text(text: str, max_length: int = 1000) -> List[str]:
    """Split text into chunks while trying to preserve logical boundaries."""
    # Split by double newlines first to preserve document structure
    sections = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_length = 0

    for section in sections:
        # If section is too long, split it by single newlines
        if len(section) > max_length:
            lines = section.split("\n")
            for line in lines:
                if current_length + len(line) > max_length and current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                current_chunk.append(line)
                current_length += len(line) + 1  # +1 for newline
        else:
            # If adding this section would exceed max_length, start a new chunk
            if current_length + len(section) > max_length and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.append(section)
            current_length += len(section) + 2  # +2 for double newline

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


def ingest_and_index_package(package_name_or_repo_url: str):
    # Use the last part of the URL or package name as the identifier
    identifier = package_name_or_repo_url.split("/")[-1].replace(".git", "")
    db_dir = f".chroma_{identifier}"
    if os.path.exists(db_dir):
        console.print(f"[yellow]Package already indexed at {db_dir}[/]")
        return  # Already indexed

    with tempfile.TemporaryDirectory() as tmpdir:
        package_dir = download_and_extract_package(package_name_or_repo_url, tmpdir)
        files = collect_text_files(package_dir)

        if not files:
            raise RuntimeError(
                f"No valid Python files found in {package_name_or_repo_url}. Please check if the package/repository contains any Python source files or documentation."
            )

        console.print("\n[bold blue]Processing files and creating chunks...[/]")
        docs = []
        metadatas = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Creating chunks...", total=len(files))

            for file_info in files:
                for chunk in chunk_text(file_info["content"]):
                    docs.append(chunk)
                    metadatas.append(
                        {
                            "source": file_info["relative_path"],
                            "type": file_info["type"],
                            "full_path": file_info["path"],
                        }
                    )
                progress.update(task, advance=1)

        console.print(f"\n[bold blue]Indexing {len(docs)} chunks with ChromaDB...[/]")

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

        console.print(f"\n[bold green]Indexing complete![/]")
        console.print(f"Total chunks indexed: {len(docs)}")
        console.print(f"Index location: {db_dir}")
