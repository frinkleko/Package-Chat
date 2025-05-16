# Package Chat

A Retrieval-Augmented Generation (RAG) bot that helps answer questions about Python packages by analyzing their documentation and source code.

## Installation

```bash
pip install -r requirements.txt
# or use uv
uv sync
```

## Environment Variables

Create a `.env` file with your configuration:

```env
# Required
OPENAI_API_KEY=your_api_key_here

# Optional
OPENAI_API_ENDPOINT=https://api.openai.com/v1  # Optional, defaults to https://api.openai.com/v1
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002  # Optional, defaults to text-embedding-ada-002
MAX_HISTORY=10  # Optional, defaults to 10
```

## Usage

### Command Line Interface

Run the bot interactively:

```bash
python rag_bot.py pandas
```

Or specify a GitHub repository:

```bash
python rag_bot.py https://github.com/pandas-dev/pandas
```

### Programmatic Usage

You can also use the bot programmatically in your Python code:

```python
from conversation import ConversationManager
from retriever import retrieve_relevant_chunks
from openai import OpenAI
import os

# Initialize components
conversation = ConversationManager(max_history=int(os.getenv("MAX_HISTORY", 10)))
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_ENDPOINT")
)

# Example usage
package_name = "pandas"
question = "How do I read a CSV file?"

# Get relevant context
context_chunks = retrieve_relevant_chunks(question, package_name)
context = "\n\n".join(context_chunks)

# Add user question and format prompt in one step
_, messages = conversation.add_message(
    role="user",
    content=question,
    context=context,
    package_name=package_name
)

# Get response from OpenAI
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    max_tokens=512,
    temperature=0.2,
)
answer = response.choices[0].message.content.strip()

# Add assistant's response to history
conversation.add_message("assistant", answer)

print(f"Answer: {answer}")
```

## How It Works

1. The bot first ingests and indexes the specified package's documentation and source code
2. When a question is asked, it:
   - Retrieves relevant context from the indexed documentation
   - Formats the conversation history and context
   - Sends the query to the language model
   - Returns the response with relevant documentation references