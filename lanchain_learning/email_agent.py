from typing import Literal
from uuid import uuid4
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy, interrupt
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver

class EmailClassifier(TypedDict):
    topic: str
    summary: str
    urgency: Literal["low", "medium", "high"]
    intent: Literal["question", "bug", "billing", "future", "complex"]

class EmailAgentState(TypedDict):
    email_content: str
    sender_email: str
    email_id: str

    classification: EmailClassifier | None

    search_results: list[str] | None
    customer_history: dict[str, any] | None

    draft_response: str | None
    messages: list[str] | None

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) 

def read_email(state: EmailAgentState) -> dict[str, any]:
    """Extract and parse email content"""
    return {
        "messages": [HumanMessage(content=f"Read the email and extract the following information: {state['email_content']}")]
    }

def classify_intent(state: EmailAgentState) -> Command[Literal["search_documentation", "human_review", "draft_response", "bug_tracking"]]:
    """Classify the email into a topic, summary, urgency, and intent"""
    llm_structured = llm.with_structured_output(EmailClassifier)
    classification_prompt = f"""
    Analyze this customer email and classify it:

    Email: {state['email_content']}
    From: {state['sender_email']}

    Provide classification including intent, urgency, topic, and summary.
    """

    classification = llm_structured.invoke(classification_prompt)
    if classification["intent"] == "billing" or classification["urgency"] == "payment":
        goto = 'human_review'
    elif classification["intent"] in ["question", "future"]:
        goto = 'search_documentation'
    elif classification["intent"] == "bug":
        goto = 'bug_tracking'
    else:
        goto = 'draft_response'
    return Command(update={"classification": classification}, goto=goto)

def search_documentation(state: EmailAgentState) -> Command[Literal["draft_response"]]:
    """Search the documentation for the email"""
    classification = state.get("classification", {})
    search_results = ["Found url documentation on {classification['topic']}", "Found web page documentation on {classification['topic']}"]
    return Command(update={"search_results": search_results}, goto="draft_response")

def bug_tracking(state: EmailAgentState) -> Command[Literal["draft_response"]]:
    """Track the bug"""
    ticket_id = str(uuid4())
    return Command(update={"search_results": [f"Created bug tracking ticket {ticket_id}"]}, goto="draft_response")

def draft_response(state: EmailAgentState) -> Command[Literal["human_review", "send_reply"]]:
    """Draft the response"""
    classification = state.get("classification", {})

    context_sections = []
    if state.get("search_results"):
        formated_docs  = "\n".join([f" -{doc}" for doc in state.get("search_results")])
        context_sections.append(f"Documentation found: {formated_docs}")

    draft_prompt = f"""
    Draft a response to this customer email:
    {state['email_content']}

    Email intent: {classification.get('intent', 'unknown')}
    Urgency level: {classification.get('urgency', 'medium')}

    {chr(10).join(context_sections)}

    Guidelines:
    - Be professional and helpful
    - Address their specific concern
    - Use the provided documentation when relevant
    """    

    response = llm.invoke(draft_prompt)

    needs_review = (
        classification.get("urgency") in ["high", "critical"] or
        classification.get("intent") == "complex"
    )

    goto = "human_review" if needs_review else "send_reply"
    return Command(update={"draft_response": response.content}, goto=goto)

def human_review(state: EmailAgentState) -> Command[Literal["send_reply", END]]:
    """Pause for human review using interrupt and route based on decision"""

    classification = state.get('classification', {})
    human_decision = interrupt({
        "email_id": state.get('email_id',''),
        "original_email": state.get('email_content',''),
        "draft_response": state.get('draft_response',''),
        "urgency": classification.get('urgency'),
        "intent": classification.get('intent'),
        "action": "Please review and approve/edit this response"
    })
    if(human_decision.get('approved')):
        # Get the response from either 'response' or 'edited_response' field
        response_text = human_decision.get('response') or human_decision.get('edited_response') or state.get('draft_response', '')
        return Command(update={"draft_response": response_text}, goto="send_reply")
    else:
        return Command(update={}, goto=END)


def send_reply(state: EmailAgentState) -> dict:
    """Send the email response"""
    # Integrate with email service
    draft = state.get('draft_response', '')
    if draft:
        print(f"Sending reply: {draft[:100]}...")
    else:
        print("Warning: No draft_response found, cannot send reply")
    return {}

workflow = StateGraph(EmailAgentState)
workflow.add_node("read_email", read_email)
workflow.add_node("classify_intent", classify_intent)

workflow.add_node("search_documentation", search_documentation, retry_policy=RetryPolicy(max_attempts=3))
workflow.add_node("bug_tracking", bug_tracking)
workflow.add_node("draft_response", draft_response)
workflow.add_node("human_review", human_review)
workflow.add_node("send_reply", send_reply)

workflow.add_edge(START, "read_email")
workflow.add_edge("read_email", "classify_intent")
workflow.add_edge("send_reply", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


initial_state = {
    "email_content": "I was charged twice for my subscription! This is urgent!",
    "sender_email": "customer@example.com",
    "email_id": "email_123",
    "messages": []
}

config = {"configurable": {"thread_id": "email_123"}}
result = app.invoke(initial_state, config)
# The graph will pause at human_review
if result.get('draft_response'):
    print(f"Draft ready for review: {result['draft_response'][:100]}...")
else:
    print(f"State: {list(result.keys())}")
    print("Graph paused at human_review (no draft_response yet)")

# When ready, provide human input to resume
from langgraph.types import Command

human_response = Command(
    resume={
        "approved": True,
        "edited_response": "We sincerely apologize for the double charge. I've initiated an immediate refund..."
    }
)

# Resume execution
final_result = app.invoke(human_response, config)
print(f"Email sent successfully!")