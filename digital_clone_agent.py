import os
import random
from typing import Literal, TypedDict, Set
from uuid import uuid4
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import markdown
from langchain_core.documents import Document
from langchain_core.runnables import graph
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import Command
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.prebuilt import tools_condition
from langgraph.types import interrupt
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import RecursiveUrlLoader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.firecrawl import FireCrawlLoader
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

api_key = os.getenv("FIRECRAWL_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class DigitalCloneState(TypedDict):
    user_name: str
    email: str
    documents: list[str]
    meeting_time: str
    score: float
    availability: bool
    booking_details: str
    question: str
    response: str


class Evaluation(TypedDict):
    score: float
    reason: str


def markdown_to_text(markdown_string):
    """Convert markdown to plain text"""
    html = markdown.markdown(markdown_string)
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text()


def retrive_documents(state: DigitalCloneState):
    """Retrive the documents from the database"""
    client = QdrantClient(path="qdrant_db")

    collection_name = "math_5th_grade"
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    if not client.collection_exists(collection_name=collection_name):
        print(f"Collection {collection_name} does not exist, creating it...")
        test_embedding = embeddings.embed_query("test")
        vector_size = len(test_embedding)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

        params = {
            "maxDepth": 2,
            "includePaths": [".*5th-grade.*"],
            "limit": 10,
            "formats": ["markdown"],
        }

        loader = FireCrawlLoader(
            url="https://www.fishtanklearning.org/curriculum/math/5th-grade",
            api_key=api_key,
            mode="crawl",
            params=params,
        )
        print("Starting FireCrawl...")
        try:
            docs = loader.load()
            print(f"Loaded {len(docs)} documents")
        except Exception as e:
            print(f"Error loading documents: {e}")
            return {"documents": []}

        docs = [
            Document(
                page_content=markdown_to_text(doc.page_content),
                metadata=doc.metadata,
                id=str(uuid4()),
            )
            for doc in docs
        ]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )
        all_splits = text_splitter.split_documents(docs)
        # TODO - check metatadta
        all_splits = [
            Document(
                page_content=split.page_content,
                metadata={
                    "url": split.metadata.get("url", ""),
                    "title": split.metadata.get("title", ""),
                    "description": split.metadata.get("description", ""),
                    "doc_id": split.metadata.get("id", ""),
                },
                id=str(uuid4()),
            )
            for split in all_splits
        ]
        print(
            f"Adding {len(all_splits)} documents to the collection {collection_name}..."
        )
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )
        vector_store.add_documents(all_splits)
    else:
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )

    results = vector_store.similarity_search(state["question"], k=3)
    return {"documents": results}


def generate_response(state: DigitalCloneState):
    """Generate a response"""
    with open("style_profile.txt", "r") as f:
        profile_text = f.read()

    # Format the retrieved documents
    documents_text = ""
    if state.get("documents"):
        documents_text = "\n\nRelevant context from provided documents:\n"
        for i, doc in enumerate(state["documents"], 1):
            documents_text += f"\n--- Document {i} ---\n"
            documents_text += f"{doc.page_content}\n"
            if doc.metadata:
                documents_text += f"Source: {doc.metadata.get('url', doc.metadata.get('title', 'Unknown'))}\n"

    prompt = f"""You are a grade 5 math teacher. You can ONLY answer questions about mathematics subjects taught in school (specifically 5th grade math topics). 

IMPORTANT RULES:
1. If the question is about mathematics (5th grade level), answer it using ONLY the information from the 
provided documents below. Do not use any generic knowledge or information not found in the documents.
2. If the question is about ANY other subject or topic (science, history, sports, hobbies, etc.), you MUST 
politely decline and explain that you can only help with math questions.

Question: {state["question"]}

{documents_text}

Communication style guidelines:
{profile_text}

Remember: Use ONLY the information from the provided documents above. If the documents don't contain relevant information for a math question, say so. If the question is not about math, politely decline to answer.
"""

    response = llm.invoke(prompt)
    print(f"Response: {response.content}")
    return {"response": response.content}


def evaluate_response(state: DigitalCloneState):
    """Evaluate the response"""

    eval_prompt = f"""You are an evaluation agent that scores RAG responses. 

    here is the question:
    {state["question"]}

    here is the response:
    {state["response"]}

    For each response, you must evaluate and calculate:

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

    **Output Format:**
    For each evaluation, provide:
    - StyleScore: [score between 0.0 and 1.0]
    - Groundedness: [score between 0.0 and 1.0]
    - Confidence: [score between 0.0 and 1.0]
    """

    llm_structured = llm.with_structured_output(Evaluation)
    evaluation = llm_structured.invoke(eval_prompt)
    score = (
        0.4 * evaluation["StyleScore"]
        + 0.4 * evaluation["Groundedness"]
        + 0.2 * evaluation["Confidence"]
    )
    return {"score": score}


def human_review(state: DigitalCloneState):
    """Schedule a meeting"""
    human_decision = interrupt(
        {
            "action": "schedule_meeting",
            "message": "We need to schedule a meeting with teacher. Please approve this meeting.",
            "user_name": state.get("user_name", ""),
            "email": state.get("email", ""),
            "documents": state.get("documents", []),
            "meeting_time": state.get("meeting_time", ""),
        }
    )
    if human_decision.get("approved"):
        return {
            "meeting_time": human_decision.get("meeting_time"),
            "user_name": human_decision.get("user_name"),
            "email": human_decision.get("email"),
        }
    else:
        return {"meeting_time": None, "user_name": None, "email": None}


def get_availability(state: DigitalCloneState):
    """Get the availability of the teacher"""
    random_availability = random.choice([True, False])
    return {"availability": random_availability}


def book_meeting(state: DigitalCloneState):
    """Book a meeting"""
    return {
        "booking_details": f"Meeting booked for {state.get('meeting_time')} with {state.get('user_name')} at {state.get('email')}"
    }


def check_score(state: DigitalCloneState):
    """Check the score"""
    if state.get("score") < 0.5:
        return "schedule_meeting"
    else:
        return "end"


def check_availability(state: DigitalCloneState):
    """Check the availability"""
    if state.get("availability"):
        return "book_meeting"
    else:
        return "ask_for_new_time"


def ask_for_new_time(state: DigitalCloneState):
    """Ask for a new time"""
    new_time = interrupt(
        {
            "action": "ask_new_time",
            "message": "The teacher is not available at the requested time. Please provide a new meeting time.",
            "user_name": state.get("user_name", ""),
            "email": state.get("email", ""),
            "documents": state.get("documents", []),
            "meeting_time": state.get("meeting_time", "")
        }
    )
    if new_time.get("reject"):
        return {"meeting_time": None}
    else:
        return {"meeting_time": new_time.get("new_time")}


def route_after_new_time(state: DigitalCloneState):
    """Route after asking for new time"""
    if state.get("meeting_time"):
        return "book_meeting"
    else:
        return "end"


workflow = StateGraph(DigitalCloneState)
workflow.add_node("retrive_documents", retrive_documents)
workflow.add_node("evaluate_response", evaluate_response)
workflow.add_node("get_availability", get_availability)
workflow.add_node("human_review", human_review)
workflow.add_node("ask_for_new_time", ask_for_new_time)
workflow.add_node("book_meeting", book_meeting)
workflow.add_node("generate_response", generate_response)

workflow.add_edge(START, "retrive_documents")
workflow.add_edge("retrive_documents", "generate_response")
workflow.add_edge("generate_response", "evaluate_response")
workflow.add_conditional_edges(
    "evaluate_response", check_score, {"schedule_meeting": "human_review", "end": END}
)
workflow.add_conditional_edges(
    "human_review",
    lambda state: "get_availability" if state.get("meeting_time") else "end",
    {"get_availability": "get_availability", "end": END},
)
workflow.add_conditional_edges(
    "get_availability",
    lambda state: "book_meeting" if state.get("availability") else "ask_for_new_time",
    {"book_meeting": "book_meeting", "ask_for_new_time": "ask_for_new_time"},
)
workflow.add_conditional_edges(
    "ask_for_new_time",
    route_after_new_time,
    {"book_meeting": "book_meeting", "end": END},
)
workflow.add_edge("book_meeting", END)

graph = workflow.compile(checkpointer=MemorySaver())
graph_image = graph.get_graph(xray=True).draw_mermaid_png()
with open("digital_clone_visualization.png", "wb") as f:
    f.write(graph_image)
print("Graph visualization saved to 'digital_clone_visualization.png'")

config = {"configurable": {"thread_id": "digital_clone_1"}}
result = graph.invoke(
    {"question": "My son struggles with tennis shots, what should I do?"}, config=config
)
print(result)


if "__interrupt__" in result:
    print("\n" + "="*50)
    print("HUMAN REVIEW REQUIRED")
    print("="*50)
    interrupt_data = result["__interrupt__"][0].value
    action = interrupt_data.get('action', 'unknown')
    message = interrupt_data.get('message', 'Review required')
    
    print(f"\nAction: {action}")
    print(f"Message: {message}")
    print(f"\nDocuments: {len(interrupt_data.get('documents', []))} documents retrieved")
    
    # Handle different interrupt types
    if action == "schedule_meeting":
        # Prompt for human input for scheduling meeting
        print("\nPlease provide your decision:")
        approved = input(f"{message} (yes/no): ").lower().strip() == "yes"
        
        if approved:
            user_name = input("Enter user name: ").strip()
            email = input("Enter email: ").strip()
            meeting_time = input("Enter meeting time: ").strip()
            
            # Resume with human input
            human_response = Command(
                resume={
                    "approved": True,
                    "user_name": user_name,
                    "email": email,
                    "meeting_time": meeting_time
                }
            )
        else:
            human_response = Command(
                resume={
                    "approved": False
                }
            )
    
    elif action == "ask_new_time":
        # Prompt for new time
        print("\nPlease provide a new meeting time:")
        new_time = input(f"{message}\nEnter new meeting time (or 'reject' to cancel): ").strip()
        
        if new_time.lower() == "reject":
            human_response = Command(
                resume={
                    "reject": True
                }
            )
        else:
            human_response = Command(
                resume={
                    "reject": False,
                    "new_time": new_time
                }
            )
    
    else:
        # Fallback for unknown interrupt types
        print(f"\nUnknown interrupt action: {action}")
        human_response = Command(resume={})
    
    # Resume execution
    final_result = graph.invoke(human_response, config=config)
    print("\nFinal result:", final_result)
    
    # Check if there's another interrupt (for chained interrupts)
    while "__interrupt__" in final_result:
        interrupt_data = final_result["__interrupt__"][0].value
        action = interrupt_data.get('action', 'unknown')
        message = interrupt_data.get('message', 'Review required')
        
        print("\n" + "="*50)
        print("HUMAN REVIEW REQUIRED (Continued)")
        print("="*50)
        print(f"\nAction: {action}")
        print(f"Message: {message}")
        
        if action == "schedule_meeting":
            approved = input(f"{message} (yes/no): ").lower().strip() == "yes"
            if approved:
                user_name = input("Enter user name: ").strip()
                email = input("Enter email: ").strip()
                meeting_time = input("Enter meeting time: ").strip()
                human_response = Command(
                    resume={
                        "approved": True,
                        "user_name": user_name,
                        "email": email,
                        "meeting_time": meeting_time
                    }
                )
            else:
                human_response = Command(resume={"approved": False})
        
        elif action == "ask_new_time":
            new_time = input(f"{message}\nEnter new meeting time (or 'reject' to cancel): ").strip()
            if new_time.lower() == "reject":
                human_response = Command(resume={"reject": True})
            else:
                human_response = Command(resume={"reject": False, "new_time": new_time})
        else:
            human_response = Command(resume={})
        
        final_result = graph.invoke(human_response, config=config)
        print("\nFinal result:", final_result)
else:
    print("Result:", result)
