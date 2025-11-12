from planning_agent import planning_agent

query = "My daughter won't listen to me about anything. Just ell me what to do?"

print("=" * 60)
print("QUERY:")
print("=" * 60)
print(query)
print("\n" + "=" * 60)
print("PLANNING AGENT RESPONSE:")
print("=" * 60)

try:
    result = planning_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })

    print("\n" + "=" * 60)
    print("FINAL RESPONSE:")
    print("=" * 60)
    if isinstance(result, dict) and "messages" in result:
        final_message = result["messages"][-1]
        if hasattr(final_message, "content"):
            print(final_message.content)
        elif hasattr(final_message, "text"):
            print(final_message.text)
        else:
            print(str(final_message))
    else:
        print(str(result))
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("=" * 60)

