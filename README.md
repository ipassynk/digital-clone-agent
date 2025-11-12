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

## Cal integration
The agent is able to call cal.com and book appointments.

Example output:
```
The team meeting for John Doe has been scheduled for tomorrow at 7pm. The meeting URL is https://app.cal.com/video/cYCtPagE3D6v5x9XTJZurN, and the location is also available on this link. The meeting duration is 30 minutes, and it will take place from 7pm to 7:30pm tomorrow night.
```


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

## TODO

- Switch to Qdrant vector database
- Improve calendar integration (support recurring events, different calendar providers)
- Add a chatbot frontend
- Enhance error handling and logging for all agents
- Add unit and integration tests
- Parameterize model, embedding, and threshold settings via config file or environment variables
- Support for multi-user sessions
- Add schema for Eval_agent output with_structured_output. Look https://docs.langchain.com/oss/python/langgraph/agentic-rag#4-grade-documents
- Switch to LangGraph
