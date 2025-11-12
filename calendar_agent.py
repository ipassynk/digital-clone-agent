from math import e
from langchain.agents import create_agent
from langchain_core.tools import tool
import os
from langchain.chat_models import init_chat_model
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import dateparser

load_dotenv()

model = init_chat_model("llama3.1:8b", model_provider="ollama")


@tool
def parse_relative_date(relative_date_str: str, time_str: str = "") -> str:
    """Parse a relative date string (like 'next Tuesday', 'tomorrow', 'next week') into an actual ISO datetime.

    MANDATORY FIRST STEP: You MUST call this tool FIRST whenever the user's request contains ANY relative date terms
    (e.g., 'next Tuesday', 'tomorrow', 'Monday', '2pm', 'next week'). 
    
    IMPORTANT: Call this tool ONLY ONCE. After receiving the result, you MUST call get_available_time_slots next.
    DO NOT call this tool multiple times. DO NOT call create_calendar_event until you have called get_available_time_slots.

    Args:
        relative_date_str: Natural language date like 'next Tuesday', 'tomorrow', 'next week', 'Monday', etc.
        time_str: Optional time string like '2pm', '14:00', '2:00 PM'. If provided, will be combined with the date.
    
    Returns:
        ISO datetime string (YYYY-MM-DDTHH:MM:SS) for the calculated date and time.
        After receiving this result, you MUST:
        1. Extract the date part (YYYY-MM-DD) from the ISO datetime
        2. Call get_available_time_slots with that date
        3. DO NOT proceed to create_calendar_event without calling get_available_time_slots first
    
    Examples:
        - parse_relative_date('next Tuesday', '2pm') -> '2025-11-18T14:00:00'
        - parse_relative_date('tomorrow', '10:00') -> '2025-11-12T10:00:00'
        - parse_relative_date('next week Monday') -> '2025-11-17T00:00:00'
    """
    print(f"[TOOL CALL] parse_relative_date")
    print(f"   relative_date_str={relative_date_str}, time_str={time_str}")

    # Combine date and time strings if both are provided
    date_time_str = relative_date_str
    if time_str:
        date_time_str = f"{relative_date_str} {time_str}"
    
    parsed_date = dateparser.parse(date_time_str)
    res = parsed_date.isoformat()
    print(f"[TOOL RESULT] parse_relative_date {res}")
    return res

@tool
def create_calendar_event(start_time: str, name: str, email: str, description: str = "") -> str:
    """Create a calendar event. Requires exact ISO datetime format (YYYY-MM-DDTHH:MM:SS).

    CRITICAL REQUIREMENTS - YOU MUST FOLLOW THIS EXACT SEQUENCE (ORDER IS CRITICAL):
    
    CRITICAL ORDER: parse_relative_date -> get_available_time_slots -> create_calendar_event
    DO NOT call this tool (create_calendar_event) before get_available_time_slots. This will cause a 400 error.
    
    STEP 1: If the request contains ANY relative date terms (e.g., 'next Tuesday', 'tomorrow', 'next week', 'Monday', '2pm'),
            you MUST FIRST call parse_relative_date to convert it to an ISO datetime. DO NOT pass relative dates here.
    
    STEP 2: You MUST call get_available_time_slots FIRST to verify the requested time slot is available.
            Extract the date part (YYYY-MM-DD) from the ISO datetime and call get_available_time_slots with it.
            THIS STEP IS MANDATORY - DO NOT SKIP IT.
            DO NOT call create_calendar_event before this step - this is WRONG and will cause a 400 error.
    
    STEP 3: Check the response from get_available_time_slots. Verify that the requested time slot (from parse_relative_date)
            is actually available in the returned list of available slots.
    
    STEP 4: IF the time slot is confirmed available, YOU MUST IMMEDIATELY call this tool (create_calendar_event).
            THIS IS MANDATORY - DO NOT just say the event is scheduled without calling this tool.
            The event will NOT be created unless you actually call create_calendar_event.
            YOU MUST CALL THIS TOOL - it is not optional if the slot is available.
            BUT: Only call this tool AFTER completing steps 1, 2, and 3. DO NOT call it before get_available_time_slots.
    
    The start_time parameter MUST be in ISO 8601 format (YYYY-MM-DDTHH:MM:SS), NOT a relative date string.
    If you receive a relative date like 'next Tuesday at 2pm', you MUST call parse_relative_date first.
    
    DO NOT call this tool without first calling get_available_time_slots. This will cause errors.
    DO NOT skip calling this tool after verifying availability - the event must be created via this tool call.
    """
    print(f"[TOOL CALL] create_calendar_event {start_time} {name} {email} {description}")
    api_key = os.getenv("CAL_API_KEY")
    event_type_id = os.getenv("CAL_EVENT_TYPE_ID")  
    USER_TIMEZONE = "America/Toronto"
    
    book_url = "https://api.cal.com/v2/bookings"
   
   
    if start_time.endswith('Z'):
        start_datetime = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
    else:
        # If no timezone, assume it's in the user's timezone and convert to UTC
        start_datetime = datetime.fromisoformat(start_time)
        if start_datetime.tzinfo is None:
            # Naive datetime - assume it's in user timezone, convert to UTC
            local_tz = ZoneInfo(USER_TIMEZONE)
            start_datetime = start_datetime.replace(tzinfo=local_tz)
    
    # Convert to UTC and format with Z suffix
    start_datetime_utc = start_datetime.astimezone(timezone.utc)
    start_iso = start_datetime_utc.isoformat().replace('+00:00', 'Z')
    
    payload = {
        "eventTypeId": int(event_type_id),
        "start": start_iso,
        "attendee": {
            "email": email,
            "name": name,
            "timeZone": USER_TIMEZONE
        },
        "metadata": {
            "title": "Booking from calendar agent",
            "description": description
        }
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "cal-api-version": "2024-08-13",
        "Content-Type": "application/json"
    }
    print(f"   → Request payload: {payload}")

    resp = requests.post(book_url, headers=headers, json=payload)
    ret = resp.json()
    
    print(f"   → Response status: {resp.status_code}")
    print(f"   → Response body: {ret}")
    print("--------------------------------\n\n")
    
    # If there's an error, include it in the return value so the agent can see it
    if resp.status_code >= 400:
        error_msg = ret.get('error', {}).get('message', 'Unknown error')
        return f"ERROR: Failed to create calendar event. Status {resp.status_code}: {error_msg}. Response: {ret}"
    
    return ret


@tool
def get_available_time_slots(date: str) -> list[str]:
    """Check calendar availability on a specific date.

    CRITICAL ORDER REQUIREMENT: This tool MUST be called BEFORE create_calendar_event. 
    The correct order is: parse_relative_date -> get_available_time_slots -> create_calendar_event
    DO NOT call create_calendar_event before this tool. This is WRONG and will cause a 400 error.
    It is MANDATORY and CANNOT be skipped. You MUST call this tool after parse_relative_date and BEFORE create_calendar_event in ALL cases.
    
    YOU CANNOT KNOW IF A TIME SLOT IS AVAILABLE WITHOUT CALLING THIS TOOL.
    DO NOT assume availability. DO NOT say "the time slot is available" without calling this tool.
    DO NOT call create_calendar_event before calling this tool - this is WRONG.
    
    IMPORTANT: This tool ONLY accepts ONE parameter: 'date'. DO NOT pass any other parameters like name, email, description, or start_time.
    Those parameters are ONLY for create_calendar_event, NOT for this tool.
    
    CRITICAL: The 'date' parameter MUST be the ACTUAL RESULT returned by parse_relative_date, NOT a placeholder string.
    You MUST use the exact ISO datetime string returned by parse_relative_date (e.g., '2025-11-12T19:00:00').
    DO NOT use placeholder text like 'ISO datetime string from previous step' - you MUST use the actual result value.
    
    Args:
        date: Date in ISO format (YYYY-MM-DDTHH:MM:SS) or (YYYY-MM-DD). Must be an actual date, not relative terms like "next Tuesday".
              If you receive a relative date, use parse_relative_date to convert it first.
              Example: Pass '2025-11-12T19:00:00' or '2025-11-12' - ONLY this parameter, nothing else.
              CRITICAL: This MUST be the actual result from parse_relative_date, not a placeholder or description.

    Returns:
        A list of available time slots for the specified date. You MUST check this list to verify
        if the requested time slot is available before calling create_calendar_event.

    WORKFLOW REQUIREMENT - FOLLOW THIS EXACT SEQUENCE (ORDER IS CRITICAL):
    1. Call parse_relative_date first (if relative dates are present) - get ISO datetime like '2025-11-13T14:00:00'
    2. Extract the ACTUAL RESULT from parse_relative_date (e.g., '2025-11-13T14:00:00') - DO NOT use placeholder text
    3. Call THIS tool (get_available_time_slots) with ONLY the date parameter from step 2 - THIS IS MANDATORY
       Example: get_available_time_slots(date='2025-11-13T14:00:00') - use the ACTUAL result, not a description
       DO NOT pass name, email, description, or start_time - those are for create_calendar_event only
       DO NOT call create_calendar_event before this step - this is WRONG and will cause a 400 error
    4. Wait for the tool to return the list of available slots
    5. Check if the requested time slot (from step 1) is in the returned available slots
    6. ONLY THEN call create_calendar_event if the slot is available (AFTER completing steps 1-5)
    
    CRITICAL ORDER: parse_relative_date -> get_available_time_slots -> create_calendar_event
    DO NOT skip this step. DO NOT call create_calendar_event without calling this tool first.
    DO NOT call create_calendar_event before this tool - this is WRONG and will cause a 400 error.
    DO NOT assume availability - you MUST call this tool to check.
    DO NOT use placeholder strings - you MUST use the actual result from parse_relative_date.
    """
    print(f"[TOOL CALL] get_available_time_slots on {date}")

    api_key = os.getenv("CAL_API_KEY")
    username = os.getenv("CAL_USERNAME")
    event_type_id = os.getenv("CAL_EVENT_TYPE_ID")

    # Parse the date/datetime and convert to UTC ISO format
    USER_TIMEZONE = "America/Toronto"
    
    # Validate that date is not a placeholder string
    if not date or date.strip() in ['ISO datetime string from previous step', 'date from previous step', 'result from previous step']:
        error_msg = f"ERROR: Invalid date parameter. You must pass the ACTUAL result from parse_relative_date, not a placeholder string. Received: '{date}'. Please call parse_relative_date first and use its exact return value."
        print(f"   → {error_msg}")
        return [error_msg]
    
    try:
        if 'T' in date:
            # If datetime provided, parse it
            if date.endswith('Z'):
                start_datetime = datetime.fromisoformat(date.replace('Z', '+00:00'))
            else:
                # Assume datetime is in user's timezone if no timezone info
                start_datetime = datetime.fromisoformat(date)
                if start_datetime.tzinfo is None:
                    # If no timezone, we'll treat as naive and add UTC (API expects UTC)
                    start_datetime = start_datetime.replace(tzinfo=timezone.utc)
        else:
            # If only date provided, use start of day in UTC
            start_datetime = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
    except ValueError as e:
        error_msg = f"ERROR: Invalid date format '{date}'. The date parameter must be a valid ISO datetime string (e.g., '2025-11-12T19:00:00') or ISO date (e.g., '2025-11-12'). You must use the ACTUAL result from parse_relative_date, not a placeholder. Error: {str(e)}"
        print(f"   → {error_msg}")
        return [error_msg]
    
    # Calculate end datetime (next day)
    end_datetime = start_datetime + timedelta(days=1)
    
    # Convert to UTC ISO format strings with Z suffix
    if start_datetime.tzinfo:
        start_iso = start_datetime.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
        end_iso = end_datetime.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
    else:
        start_iso = start_datetime.isoformat() + 'Z'
        end_iso = end_datetime.isoformat() + 'Z'

    url = "https://api.cal.com/v2/slots"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "cal-api-version": "2024-09-04"
    }
    params = {
        "eventTypeId": int(event_type_id),
        "start": start_iso,
        "end": end_iso
    }

    print(f"   → Request params: {params}")
    response = requests.get(url, headers=headers, params=params)
    res = response.json()
    print(f"   → Result: {res}")
    print("--------------------------------\n\n")
    return res


CALENDAR_AGENT_PROMPT = (
    "You are a calendar scheduling assistant. "
    "YOU MUST FOLLOW THESE STEPS IN EXACT ORDER - DO NOT SKIP ANY STEP:\n\n"
    "CRITICAL ORDER REQUIREMENT: You MUST call tools in this exact sequence:\n"
    "1. parse_relative_date (if relative dates present)\n"
    "2. get_available_time_slots (MANDATORY - must come BEFORE create_calendar_event)\n"
    "3. create_calendar_event (ONLY after steps 1 and 2 are complete)\n"
    "DO NOT call create_calendar_event before get_available_time_slots. This will cause errors.\n\n"
    "STEP 1 - PARSE RELATIVE DATES (MANDATORY IF REQUEST CONTAINS RELATIVE DATES):\n"
    "   If the user's request contains ANY relative date terms such as:\n"
    "   - 'next Tuesday', 'next week', 'tomorrow', 'Monday', 'Friday'\n"
    "   - '2pm', '10:00 AM', '14:30', or any time expressions\n"
    "   - ANY combination of day names, relative terms, or time expressions\n"
    "   \n"
    "   YOU MUST IMMEDIATELY call parse_relative_date tool FIRST (ONLY ONCE).\n"
    "   DO NOT attempt to calculate dates yourself.\n"
    "   DO NOT call get_available_time_slots or create_calendar_event until you have called parse_relative_date.\n"
    "   DO NOT call create_calendar_event before get_available_time_slots - this is WRONG and will fail.\n"
    "   \n"
    "   Example: If user says 'next Tuesday at 2pm', you MUST call:\n"
    "   parse_relative_date('next Tuesday', '2pm') -> returns ISO datetime like '2025-11-18T14:00:00'\n"
    "   \n"
    "STEP 2 - CHECK AVAILABILITY (MANDATORY - CANNOT BE SKIPPED - MUST COME BEFORE create_calendar_event):\n"
    "   After receiving the ISO datetime from parse_relative_date (e.g., '2025-11-12T19:00:00'):\n"
    "   \n"
    "   CRITICAL: You MUST call get_available_time_slots BEFORE create_calendar_event.\n"
    "   DO NOT call create_calendar_event before get_available_time_slots. This is WRONG and will cause a 400 error.\n"
    "   The correct order is: parse_relative_date -> get_available_time_slots -> create_calendar_event\n"
    "   \n"
    "   CRITICAL: You MUST use the ACTUAL RESULT VALUE returned by parse_relative_date, NOT a placeholder or description.\n"
    "   When parse_relative_date returns '2025-11-12T19:00:00', you MUST use that exact string value.\n"
    "   DO NOT use placeholder text like 'ISO datetime string from previous step' - you MUST use the actual result.\n"
    "   \n"
    "   1. Extract the ACTUAL RESULT from parse_relative_date (e.g., if it returned '2025-11-12T19:00:00', use that exact string).\n"
    "   2. YOU MUST IMMEDIATELY CALL get_available_time_slots tool with ONLY the date parameter using the ACTUAL result.\n"
    "      Example: If parse_relative_date returned '2025-11-12T19:00:00', call: get_available_time_slots(date='2025-11-12T19:00:00')\n"
    "      CRITICAL: Pass ONLY the 'date' parameter. DO NOT pass name, email, description, or start_time.\n"
    "      CRITICAL: Use the ACTUAL result value from parse_relative_date, not a placeholder or description.\n"
    "      Those parameters are ONLY for create_calendar_event, NOT for get_available_time_slots.\n"
    "      THIS IS MANDATORY - YOU CANNOT PROCEED WITHOUT THIS CALL.\n"
    "      DO NOT assume availability. DO NOT say 'the time slot is available' without calling the tool.\n"
    "      YOU MUST ACTUALLY CALL get_available_time_slots - this is not optional.\n"
    "   3. Wait for the tool to return the list of available time slots.\n"
    "   \n"
    "   CRITICAL: You CANNOT know if a time slot is available without calling get_available_time_slots.\n"
    "   DO NOT skip this step. DO NOT assume availability. DO NOT call create_calendar_event before this step.\n"
    "   IF YOU SKIP THIS STEP OR CALL create_calendar_event BEFORE THIS STEP, THE SCHEDULING WILL FAIL.\n"
    "   \n"
    "STEP 3 - VERIFY AVAILABILITY:\n"
    "   Check if the requested time slot (from parse_relative_date) is in the list of available slots returned by get_available_time_slots.\n"
    "   Note: Available slots are returned in UTC. The requested time from parse_relative_date may need timezone conversion.\n"
    "   If the time slot is NOT available, suggest alternative dates/times to the user.\n"
    "   DO NOT call create_calendar_event if the slot is unavailable.\n\n"
    "STEP 4 - CREATE EVENT (MANDATORY IF AVAILABLE - MUST COME AFTER get_available_time_slots):\n"
    "   CRITICAL: This step MUST come AFTER Step 2 (get_available_time_slots).\n"
    "   DO NOT call create_calendar_event before get_available_time_slots. This is WRONG and will cause a 400 error.\n"
    "   The correct order is: parse_relative_date -> get_available_time_slots -> create_calendar_event\n"
    "   \n"
    "   IF the time slot IS confirmed available from Step 3, YOU MUST IMMEDIATELY call create_calendar_event.\n"
    "   THIS IS MANDATORY - DO NOT just say 'the event is scheduled' without actually calling the tool.\n"
    "   YOU MUST CALL create_calendar_event with ALL required parameters:\n"
    "   - start_time: The FULL ISO datetime (YYYY-MM-DDTHH:MM:SS) from parse_relative_date (e.g., '2025-11-13T14:00:00')\n"
    "   - name: The user's name from the request (e.g., 'John Doe')\n"
    "   - email: The user's email from the request (e.g., 'joe.doe@example.com')\n"
    "   - description: The description/title from the request (e.g., 'Unresolved issues with kids')\n"
    "   \n"
    "   AFTER calling create_calendar_event, you MUST check the response:\n"
    "   - If the response contains 'ERROR' or indicates a failure (status 400, 500, etc.), DO NOT say the event was scheduled.\n"
    "   - Report the error to the user and explain what went wrong.\n"
    "   - Only confirm success if the response indicates the event was created successfully.\n"
    "   \n"
    "   CRITICAL: After verifying availability in Step 3, YOU MUST CALL create_calendar_event. DO NOT skip this step.\n"
    "   DO NOT just confirm the booking without calling the tool. The event will NOT be created unless you call create_calendar_event.\n"
    "   DO NOT call create_calendar_event without first completing Steps 2 and 3.\n"
    "   DO NOT call create_calendar_event before get_available_time_slots - this is WRONG.\n"
    "   DO NOT claim success if the tool returns an error - check the response first.\n"
    "   DO NOT use 'subject' parameter - use 'description' instead.\n"
    "   DO NOT omit any required parameters (start_time, name, email, description).\n\n"
    "STEP 5 - CONFIRM:\n"
    "   Always confirm what was scheduled in your final response.\n\n"
    "WORKFLOW SUMMARY - FOLLOW THIS EXACT SEQUENCE (ORDER IS CRITICAL):\n"
    "1. Call parse_relative_date ONCE (if relative dates present) -> returns ISO datetime like '2025-11-12T19:00:00'\n"
    "   CRITICAL: Extract the ACTUAL RESULT VALUE (e.g., '2025-11-12T19:00:00') - DO NOT use placeholder text\n"
    "2. Call get_available_time_slots with the ACTUAL date result from step 1 (MANDATORY - MUST COME BEFORE create_calendar_event)\n"
    "   Example: If step 1 returned '2025-11-12T19:00:00', call get_available_time_slots(date='2025-11-12T19:00:00')\n"
    "   CRITICAL: Use the ACTUAL result value, not a placeholder like 'ISO datetime string from previous step'\n"
    "   DO NOT call create_calendar_event before this step - this is WRONG and will cause a 400 error\n"
    "3. Verify availability from step 2 result - check if requested time slot is in the available slots list\n"
    "4. Call create_calendar_event (MANDATORY if available - MUST COME AFTER step 2 - with ALL parameters: start_time, name, email, description)\n"
    "   Use the ACTUAL result from step 1 for start_time (e.g., '2025-11-12T19:00:00')\n"
    "   YOU MUST ACTUALLY CALL THE TOOL - DO NOT just say it's scheduled without calling create_calendar_event\n"
    "   DO NOT call create_calendar_event before get_available_time_slots - this is WRONG\n"
    "5. Confirm the booking\n\n"
    "CRITICAL ORDER: parse_relative_date -> get_available_time_slots -> create_calendar_event\n"
    "NEVER call create_calendar_event before get_available_time_slots. This will cause a 400 error.\n"
    "NEVER skip step 2 (get_available_time_slots). NEVER skip step 4 (create_calendar_event) if slot is available.\n"
    "NEVER assume availability without calling get_available_time_slots.\n"
    "NEVER say an event is scheduled without actually calling create_calendar_event.\n"
    "IF YOU SKIP STEP 2 OR CALL create_calendar_event BEFORE STEP 2, THE SCHEDULING WILL FAIL.\n"
)

calendar_agent = create_agent(
    model,
    tools=[parse_relative_date, get_available_time_slots, create_calendar_event],
    system_prompt=CALENDAR_AGENT_PROMPT,
)


@tool
def schedule_event(request: str) -> str:
    """Schedule calendar events using natural language.

    Use this when the user wants to create, modify, or check calendar appointments.
    Handles date/time parsing, availability checking, and event creation.

    Input: Natural language scheduling request (e.g., 'meeting with design team next Tuesday at 7pm for user John Doe joe.doe@example.com with title 'Team Meeting'')
    """
    print(f"[TOOL CALL] schedule_event")
    print(f"   request={request}")
    result = calendar_agent.invoke({"messages": [{"role": "user", "content": request}]})
    response = result["messages"][-1].text
    print(f"   → Result: {response}")
    return response


if __name__ == "__main__":
    # Test the calendar agent
    query = "Schedule a team meeting tomorrow at 7pm for user John Doe joe.doe@example.com with about 'Unresolved issues with kids'"
    result = schedule_event.invoke({"request": query})
    print("Result from agent:")
    print(result)
