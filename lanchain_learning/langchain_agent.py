from uuid import uuid4
from langchain_classic.output_parsers import DatetimeOutputParser
from langchain_classic.output_parsers import CommaSeparatedListOutputParser
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.graph.message import uuid
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.documents import Document

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def date_example():
    parser = DatetimeOutputParser()
    print(parser.get_format_instructions())
    prompt = PromptTemplate.from_template(
        "Answer the question.\n{format_instructions}\n{question}",
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser
    print(chain.invoke({"question": "When is iPhone 17 release date?"}))


def list_example():
    parser = CommaSeparatedListOutputParser()
    prompt = PromptTemplate.from_template(
        "Answer the question.\n{format_instructions}\n{question}",
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser
    print(chain.invoke({"question": "What are the top 3 movies?"}))


def pydantic_example():
    class Author(BaseModel):
        name: str = Field(description="The name of the author")
        number: int = Field(description="The number of books written by the author")
        books: list[str] = Field(description="The list of books written by the author")

    parser = PydanticOutputParser(pydantic_object=Author)
    prompt = PromptTemplate.from_template(
        "Answer the question.\n{format_instructions}\n{question}",
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser
    print(chain.invoke({"question": "What is the name of the author?"}))


def structured_example():
    class Author(BaseModel):
        name: str = Field(description="The name of the author")
        number: int = Field(description="The number of books written by the author")
        books: list[str] = Field(description="The list of books written by the author")

    ll_structured = llm.with_structured_output(Author)
    ret = ll_structured.invoke("Get gets written by Lev Tolstoy")
    print(ret)


def sentiment_example():
    prompt = PromptTemplate.from_template(
        "Determinate sentiment of this feedback:\n {feedback}"
    )
    chain = prompt | llm | StrOutputParser()
    print(chain.invoke({"feedback": "The product is great!"}))


def runnable_lambda_example():
    parser_template = PromptTemplate.from_template(
        "Parser and clean the following customer feedback:\n {feedback}"
    )
    summary_template = PromptTemplate.from_template(
        "Summarize the following customer feedback:\n {cleaned_feedback}"
    )

    chain = (
        parser_template
        | llm
        | StrOutputParser()
        | RunnableLambda(lambda x: {"cleaned_feedback": x})
        | summary_template
        | llm
        | StrOutputParser()
    )
    print(
        chain.invoke(
            {
                "feedback": "The delivery was late and I din't like the prouct and weather is shitty and blalal"
            }
        )
    )


def runnable_function_example():
    def clean_feedback(x: str) -> str:
        print(f"Cleaning feedback: {x}")
        return {"cleaned_feedback": x}

    parser_template = PromptTemplate.from_template(
        "Parser and clean the following customer feedback:\n {feedback}"
    )
    summary_template = PromptTemplate.from_template(
        "Summarize the following customer feedback:\n {cleaned_feedback}"
    )

    chain = (
        parser_template
        | llm
        | StrOutputParser()
        | RunnableLambda(lambda x: clean_feedback(x))
        | summary_template
        | llm
        | StrOutputParser()
    )
    print(
        chain.invoke(
            {
                "feedback": "The delivery was late and I din't like the prouct and weather is shitty and blalal"
            }
        )
    )

def feedback_response_example():
    thank_you_template = PromptTemplate.from_template(
        "Giving feedback draft a thank you letter.\n{feedback}."
    )

    nutural_template = PromptTemplate.from_template(
        "Giving feedback draft a letter asking to provide more details.\n{feedback}."
    )

    angry_template = PromptTemplate.from_template(
        "Giving feebak draft an applogy letter.\n{feedback}."
    )

    thank_you_chain = thank_you_template | llm | StrOutputParser()
    nutural_chain = nutural_template | llm | StrOutputParser()
    angry_chain = angry_template | llm | StrOutputParser()


    def route(info):
        if "positive" in info["sentiment"].lower():
            return thank_you_chain
        elif "neutral" in info["sentiment"].lower():
            return nutural_chain
        elif "negative" in info["sentiment"].lower():
            return angry_chain
        else:
            return "I'm sorry, I don't know how to respond to that."


    parse_template = PromptTemplate.from_template(
        "Parse and clean the following user feedback:\n {raw_feedback}"
    )
    sentiment_template = PromptTemplate.from_template(
        "Determinate sentiment of the following user feedback and reply in one word as positive, neutral or negative:\n {parsed_feedback}"
    )
    summary_template = PromptTemplate.from_template(
        "Summarize the following user feedback in one consice sentence:\n {parsed_feedback}"
    )

    summary_chain = (
        parse_template
        | llm
        | StrOutputParser()
        | RunnableLambda(lambda x: {"parsed_feedback": x})
        | summary_template
        | llm
        | StrOutputParser()
    )
    sentiment_chain = sentiment_template | llm | StrOutputParser()

    summary = summary_chain.invoke(
        {
            "raw_feedback": "The delivery was late and I din't like the prouct and weather is shitty and blalal"
        }
    )
    print(f"summary: {summary}")
    sentiment = sentiment_chain.invoke({"parsed_feedback": summary})
    print(f"sentiment: {sentiment}")

    route_chain = {
        "feedback": lambda x: x["feedback"],
        "sentiment": lambda x: x["sentiment"],
    } | RunnableLambda(route)
    print(route_chain.invoke({"feedback": summary, "sentiment": sentiment}))


def tool_example():

    @tool
    def calculate_shipping_cost(weight: float) -> float:
        """
        Calculate the shipping cost for a given weight.
        Args:
            weight: The weight of the package in pounds
        Returns:
            The shipping cost in dollars
        """ 

        if(weight > 100):
            raise ValueError("Weight is too heavy")
        return weight * 10

    print(calculate_shipping_cost.name)
    print(calculate_shipping_cost.description)
    print(calculate_shipping_cost.args_schema)
    print(calculate_shipping_cost.invoke({"weight": 10.0}))

    tools = [calculate_shipping_cost]
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that can calculate the shipping cost for a given weight."),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)
    result = agent_executor.invoke({"input": "What is the shipping cost for a package weighing 20 kg?"})
    
    print(result["output"])
    if "intermediate_steps" in result:
        for i, (action, observation) in enumerate(result["intermediate_steps"], 1):
            print(f"\nTool Call {i}:")
            print(f"  Tool: {action.tool}")
            print(f"  Input: {action.tool_input}")
            print(f"  Result: {observation}")

def vector_store_example():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = InMemoryVectorStore(embeddings)
    docs = ["Hello, how are you?", "I am fine, thank you!", "What is your name?", "My name is John Doe."]
    docs  = [ Document(page_content=doc, id=str(uuid4())) for doc in docs]
    vector_store.add_documents(docs)
    results = vector_store.similarity_search("Hello, how are you?", k=1)
    print(results)






