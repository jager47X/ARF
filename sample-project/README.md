# ARF Sample Project — Cooking Recipe Search

A non-legal demo showing the full ARF pipeline with:
- **FAISS** as the vector database
- **Voyage AI** (`voyage-3-large`) for embeddings
- **OpenAI** (`gpt-4o-mini`) for LLM verification and summaries
- **MLP reranker** trained on made-up labeled recipe data

## Setup

```bash
# From the ARF project root
pip install -e ".[ml]"
pip install faiss-cpu voyageai openai python-dotenv
```

Ensure your `.env` file has:
```
OPENAI_API_KEY=sk-...
VOYAGE_API_KEY=pa-...
```

## Run

```bash
# Step 1: Ingest 50 recipes into FAISS
python sample-project/ingest.py

# Step 2: Train MLP reranker on labeled data
python sample-project/train.py

# Step 3: Query
python sample-project/query.py "how do I make a creamy curry?"
python sample-project/query.py "best dumpling recipe"
python sample-project/query.py "french pastry with butter"
python sample-project/query.py "spicy noodle soup"
```
