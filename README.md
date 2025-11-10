# Digital Clone Agent

Multi-agent system for intelligent query handling with automated response evaluation and conditional scheduling.

## Architecture

The system consists of four specialized agents orchestrated by a planning agent:

- **Planning Agent**: Orchestrates workflow using `retrieve_evaluate_and_decide` tool
- **RAG Agent**: Retrieves answers from knowledge base using vector search (Ollama embeddings, InMemoryVectorStore)
- **Evaluation Agent**: Scores responses on style, groundedness, and confidence (weighted: 0.4/0.4/0.2)
- **Calendar Agent**: Handles natural language scheduling requests

## Workflow

1. User query received by planning agent
2. RAG agent retrieves answer from knowledge base (`kids_issues_article.txt`)
3. Evaluation agent scores response (Final Score = 0.4 × StyleScore + 0.4 × Groundedness + 0.2 × Confidence)
4. Conditional routing:
   - Score < 0.75: Return answer to user
   - Score >= 0.75: Schedule calendar appointment

## Models

- **Planning/Calendar/Eval Agents**: Llama 3.1 8B via Ollama
- **RAG Agent**: HuggingFaceH4/zephyr-7b-beta (4-bit quantized)
- **Embeddings**: nomic-embed-text via Ollama

## Usage

```bash
python run.py
```

## Dependencies

- LangChain (agents, tools, chains)
- Ollama (local LLM inference)
- HuggingFace Transformers (RAG model)
- Vector store: InMemoryVectorStore with Ollama embeddings

## Configuration

- RAG retrieval: Top 2 documents (k=2)
- Evaluation threshold: 0.75
- Style profile: Loaded from `style_summary.txt` for personalized responses
