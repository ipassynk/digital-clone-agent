from langchain.agents import create_agent
from langchain_core.tools import tool
import os
from langchain.chat_models import init_chat_model

model = init_chat_model("llama3.1:8b", model_provider="ollama")

@tool
def create_calendar_event(
    title: str,
    start_time: str,
    end_time: str,
    attendees: list[str],
    description: str = "",
) -> str:
    """Create a calendar event. Requires exact ISO datetime format."""
    print(f"[TOOL CALL] create_calendar_event")
    print(f"   title={title}, start_time={start_time}, end_time={end_time}")
    print(f"   attendees={attendees}, description={description}")
    # Stub: In practice, this would call Google Calendar API, Outlook API, etc.
    result = f"Event created: {title} from {start_time} to {end_time} with {len(attendees)} attendees. {description}"
    print(f"   → Result: {result}")
    return result


@tool
def get_available_time_slots(
    attendees: list[str], date: str, duration_minutes: int  # ISO format: "2024-01-15"
) -> list[str]:
    """Check calendar availability for given attendees on a specific date."""
    print(f"🔧 [TOOL CALL] get_available_time_slots")
    print(f"   attendees={attendees}, date={date}, duration_minutes={duration_minutes}")
    # Stub: In practice, this would query calendar APIs
    result = ["09:00", "14:00", "16:00"]
    print(f"   → Result: {result}")
    return result


CALENDAR_AGENT_PROMPT = (
    "You are a calendar scheduling assistant. "
    "Parse natural language scheduling requests (e.g., 'next Tuesday at 2pm') "
    "into proper ISO datetime formats. "
    "Use get_available_time_slots to check availability when needed. "
    "Use create_calendar_event to schedule events. "
    "Always confirm what was scheduled in your final response."
)

calendar_agent = create_agent(
    model,
    tools=[create_calendar_event, get_available_time_slots],
    system_prompt=CALENDAR_AGENT_PROMPT,
)

@tool
def schedule_event(request: str) -> str:
    """Schedule calendar events using natural language.

    Use this when the user wants to create, modify, or check calendar appointments.
    Handles date/time parsing, availability checking, and event creation.

    Input: Natural language scheduling request (e.g., 'meeting with design team
    next Tuesday at 2pm')
    """
    print(f"[TOOL CALL] schedule_event")
    print(f"   request={request}")
    result = calendar_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    response = result["messages"][-1].text
    print(f"   → Result: {response}")
    return response

if __name__ == "__main__":
    # Test the calendar agent
    query = "Schedule a team meeting next Tuesday at 2pm for 1 hour"
    
    result = calendar_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    print("Result from agent:")
    print(result["messages"][-1].text)