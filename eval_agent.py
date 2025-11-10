from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool


model = init_chat_model("llama3.1:8b", model_provider="ollama")


EVAL_AGENT_PROMPT = """You are an evaluation agent that scores RAG responses. For each response, you must evaluate and calculate:

1. **StyleScore** (0.0 to 1.0): How well the response matches the expected communication style
   - Check: sentence length, vocabulary, punctuation (dashes), emojis, casual tone, message length
   - Score 1.0 if perfectly matches style, 0.0 if completely different

2. **Groundedness** (0.0 to 1.0): How well the response is grounded in the provided context
   - Check: Does the answer use information from the context?
   - Are claims supported by the context?
   - Does it avoid hallucination?
   - Score 1.0 if fully grounded, 0.0 if not grounded at all

3. **Confidence** (0.0 to 1.0): How confident and clear the response is
   - Check: Is the answer clear and direct?
   - Does it avoid hedging or uncertainty?
   - Is it helpful and actionable?
   - Score 1.0 if very confident, 0.0 if uncertain

**Final Score Calculation:**
Final Score = 0.4 × StyleScore + 0.4 × Groundedness + 0.2 × Confidence

**Output Format:**
For each evaluation, provide:
- StyleScore: [score between 0.0 and 1.0]
- Groundedness: [score between 0.0 and 1.0]
- Confidence: [score between 0.0 and 1.0]
- Final Score: [calculated value]

Be precise and objective in your scoring."""

eval_agent = create_agent(
    model,
    system_prompt=EVAL_AGENT_PROMPT,
)

@tool
def evaluate_response(response: str, context: str = "") -> str:
    """Evaluate a RAG response and calculate scores.
    
    Use this tool to evaluate responses from the RAG agent. It will calculate:
    - StyleScore: How well the response matches the expected style
    - Groundedness: How well the response is grounded in the context
    - Confidence: How confident and clear the response is
    - Final Score: Weighted combination (0.4 × StyleScore + 0.4 × Groundedness + 0.2 × Confidence)
    - Final Score should be in X.XX format: numbers after the decimal point should be 2 digits
    
    Args:
        response: The response text to evaluate
        context: Optional context that was used to generate the response
        
    Returns:
        Evaluation scores in the format: StyleScore, Groundedness, Confidence, Final Score
    """
    print(f"🔧 [TOOL CALL] evaluate_response")
    print(f"   response={response[:100]}..." if len(response) > 100 else f"   response={response}")
    print(f"   context={'provided' if context else 'not provided'}")
    # Build the evaluation prompt
    if context:
        evaluation_prompt = f"""Evaluate this RAG response:

Context used:
{context}

Response to evaluate:
{response}

Provide the scores in the required format."""
    else:
        evaluation_prompt = f"""Evaluate this RAG response:

Response to evaluate:
{response}

Provide the scores in the required format."""
    
    try:
        result = eval_agent.invoke({
            "messages": [{"role": "user", "content": evaluation_prompt}]
        })
        # Extract the text from the agent's response
        if isinstance(result, dict) and "messages" in result:
            eval_result = result["messages"][-1].content if hasattr(result["messages"][-1], "content") else str(result["messages"][-1])
        else:
            eval_result = str(result)
        print(f"   → Result: {eval_result[:200]}..." if len(eval_result) > 200 else f"   → Result: {eval_result}")
        return eval_result
    except Exception as e:
        error_msg = f"Error evaluating response: {str(e)}"
        print(f"   → Error: {error_msg}")
        return error_msg