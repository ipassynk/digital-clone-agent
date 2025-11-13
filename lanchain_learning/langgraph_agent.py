from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition
import time
from typing_extensions import TypedDict
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langchain_core.messages import ToolMessage, HumanMessage
import random

def add_example():
    class State(TypedDict):
        graph_state: str

    def node1(state: State) -> State:
        return {"graph_state": f"20 + {state["graph_state"]}"}

    def node2(state: State) -> State:
        result = eval(state["graph_state"])
        return {"graph_state": result}

    builder = StateGraph(State)   
    builder.add_node("node1", node1)
    builder.add_node("node2", node2)
    builder.add_edge(START, "node1")
    builder.add_edge("node1", "node2")
    builder.add_edge("node2", END)
    graph = builder.compile()
    result = graph.invoke({"graph_state": "10"})
    print(result)

def two_tools_example():
    class MyMessagesState(MessagesState):
        pass

    def temperature(location: str, date: str) -> int:
        """
        Return temperature for a given location and date
        Args:
            location: The location to get the temperature for
            date: The date to get the temperature for
        Returns:    
            The temperature for the given location and date
        """
        print(f"Getting temperature for {location} on {date}")
        return random.randint(0, 100) 

    def add(a: int, b: int) -> int:
        """Add two numbers"""
        print(f"Adding {a} and {b}")
        return a + b
        
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools([ add, temperature])

    def tool_calling_llm(state: MyMessagesState) -> MyMessagesState:
        result_with_tools = llm_with_tools.invoke(state["messages"])
        return {"messages": [result_with_tools]}

    builder = StateGraph(MyMessagesState)
    builder.add_node("tool_calling_llm", tool_calling_llm)
    builder.add_edge(START, "tool_calling_llm")
    builder.add_node("tools", ToolNode([add, temperature]))
    builder.add_conditional_edges(
        "tool_calling_llm",
        tools_condition,
        {
            "tools": "tools",
            "__end__": END
        }
    )
    builder.add_edge("tools", "tool_calling_llm")
    graph = builder.compile()
    result = graph.invoke({"messages": [HumanMessage(content="add temperature today and yesterday in San Francisco?")]})
    print(result["messages"][-1].content)


def kitchen_example():

    class MyMessagesState(MessagesState):
        pass

    def cook_meat(meat: str) -> str:
        """Cook the meat"""
        time.sleep(10)
        return f"Cooked {meat} for 10 minutes"

    def cook_vegetable(vegetable: str) -> str:
        """Cook the vegetable"""
        time.sleep(5)
        return f"Cooked {vegetable} for 5 minutes"

    def classify_food(query: str) -> str:
        """Classify the food as vegetable or meat"""
        if "tomato" in query:
            return "vegetable"
        elif "steak" in query:
            return "meat"
        else:
            return "unknown"

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools([cook_meat, cook_vegetable, classify_food])

    def tool_calling_llm(state: MyMessagesState) -> MyMessagesState:
        result_with_tools = llm_with_tools.invoke(state["messages"])
        return {"messages": [result_with_tools]}

    builder = StateGraph(MyMessagesState)
    builder.add_node("tool_calling_llm", tool_calling_llm)
    builder.add_node("tools", ToolNode([cook_meat, cook_vegetable, classify_food]))
    builder.add_node("classify_food", classify_food)
    builder.add_node("cooking_vegetable", cook_vegetable)
    builder.add_node("cooking_meat", cook_meat)

    builder.add_edge(START, "tool_calling_llm")
    builder.add_conditional_edges("tool_calling_llm", tools_condition, {
        "tools": "tools",
        "classify_food": "classify_food",
        "vegetable": "cooking_vegetable",
        "meat": "cooking_meat",
        "__end__": END
    })
    builder.add_edge("tools", "tool_calling_llm")
    builder.add_edge("cooking_vegetable", "tool_calling_llm")
    builder.add_edge("cooking_meat", "tool_calling_llm")
    builder.add_edge("tool_calling_llm", END)
    graph = builder.compile()

    graph_image = graph.get_graph(xray=True).draw_mermaid_png()
    with open("langgraph_visualization.png", "wb") as f:
        f.write(graph_image)
    print("Graph visualization saved to 'langgraph_visualization.png'")

    result = graph.invoke({"messages": [HumanMessage(content="Cook a steak and a tomato?")]})
    print(result["messages"][-1].content)

kitchen_example()
