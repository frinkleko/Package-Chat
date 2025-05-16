import argparse
from ingest import ingest_and_index_package
from retriever import retrieve_relevant_chunks
from openai import OpenAI
import os
from rich import print
from rich.prompt import Prompt
from dotenv import load_dotenv
from conversation import ConversationManager

# Load .env file if it exists
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="RAG Bot for Python Packages")
    parser.add_argument(
        "package",
        type=str,
        help="Name of the PyPI package or GitHub URL to ingest (e.g., pandas or https://github.com/user/repo)",
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            "[red]OpenAI API key required. Please set OPENAI_API_KEY in your .env file."
        )
        exit(1)

    # Initialize OpenAI client with the new API format
    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENAI_API_ENDPOINT"),
    )

    # Initialize conversation manager
    conversation = ConversationManager(max_history=int(os.getenv("MAX_HISTORY", 10)))

    print(f"[bold green]Ingesting and indexing package:[/] {args.package}")
    ingest_and_index_package(args.package)
    print(
        f"[bold green]Ready! Ask questions about {args.package}. Type 'exit' to quit."
    )

    while True:
        question = Prompt.ask("[bold blue]Your question")
        if question.strip().lower() in {"exit", "quit"}:
            break

        # Get relevant context
        context_chunks = retrieve_relevant_chunks(question)
        context = "\n\n".join(context_chunks)

        # Add user question and format prompt in one step
        _, messages = conversation.add_message(
            role="user", content=question, context=context, package_name=args.package
        )

        # Get response from OpenAI
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=512,
            temperature=0.2,
        )
        answer = response.choices[0].message.content.strip()

        # Add assistant's response to history (without context/package_name)
        conversation.add_message("assistant", answer)

        print(f"[bold green]Answer:[/] {answer}\n")


if __name__ == "__main__":
    main()
