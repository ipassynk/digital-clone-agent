from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from calendar_agent import schedule_event
from rag_agent import retrieve_and_answer
from eval_agent import evaluate_response
import re


model = init_chat_model("llama3.1:8b", model_provider="ollama")

@tool
def retrieve_evaluate_and_decide(query: str) -> str:
    """Retrieve answer, evaluate it, and decide next action.
    
    Workflow:
    1. Retrieve answer from knowledge base using RAG
    2. Evaluate the answer quality
    3. If evaluation score < 0.75: return the answer
    4. If evaluation score >= 0.75: schedule a calendar appointment
    
    IMPORTANT: Pass the COMPLETE user query exactly as received. Do not truncate or summarize it.
    Example: "My daughter won't listen to me about anything. Just ell me what to do?"
    Do not pass a truncated version of the query.

    Args:
        query: The COMPLETE user's question - pass it in full without truncation
        
    Returns:
        Either the answer (if score < 0.75) or calendar booking confirmation (if score >= 0.75)
    """
    print(f"[TOOL CALL] retrieve_evaluate_and_decide")
    print(f"   query={query}")  # Use !r to show full string with quotes
    # Step 1: Get answer from RAG
    answer = retrieve_and_answer.invoke({"query": query})
    print(f"Answer from RAG agent: {answer}")
    
    # Step 2: Evaluate the answer
    eval_result = evaluate_response.invoke({"response": answer})
    print(f"Evaluation result: {eval_result}")
    
    # Step 3: Parse the final score from evaluation result
    # Look for "Final Score: X.XX" pattern
    score_match = re.search(r'Final Score:\s*([\d.]+)', eval_result)
    if score_match:
        final_score = float(score_match.group(1))
    else:
        # Fallback: try to find any number between 0 and 1
        score_match = re.search(r'([\d.]+)', eval_result)
        final_score = float(score_match.group(1)) if score_match else 0.0
    
    # Step 4: Conditional logic
    if final_score < 0.75:
        result = f"Answer (Score: {final_score:.2f}): {answer}"
        print(f"   → Result: {result}")
        return result
    else:
        # Score >= 0.75, book appointment
        calendar_request = f"Schedule an appointment to discuss: {query}"
        result = schedule_event.invoke({"request": calendar_request})
        print(f"   → Result: {result}")
        return result

PLANNING_AGENT_PROMPT = (
    "You are a helpful personal assistant. "
    "Use the retrieve_evaluate_and_decide tool to handle user questions. "
    "This tool will retrieve answers from the knowledge base, evaluate their quality, and automatically decide whether to return the answer or schedule a calendar appointment. "
)

planning_agent = create_agent(
    model,
    tools=[retrieve_evaluate_and_decide],
    system_prompt=PLANNING_AGENT_PROMPT,
)