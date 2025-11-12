from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, json
from langchain.tools import tool
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_qdrant import FastEmbedSparse
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.documents import Document
from uuid import uuid4

with open("style_summary.txt", "r") as f:
    style_profile = f.read()


def create_vector_store():
    """Create a vector store for the kids issues article"""
    client = QdrantClient(path="qdrant_db")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="kids_issues_collection",
        embedding=embeddings,
    )

    if not client.collection_exists(collection_name="kids_issues_collection"):
        client.create_collection(
            collection_name="kids_issues_collection",
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

        loader = TextLoader("kids_issues_article.txt")
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # chunk size (characters)
            chunk_overlap=200,  # chunk overlap (characters)
            add_start_index=True,  # track index in original document
        )
        all_splits = text_splitter.split_documents(docs)
        print(f"Split blog post into {len(all_splits)} sub-documents.")
        docs = []
        for doc in all_splits:
            docs.append(Document(page_content=doc.page_content))

        uuids = [str(uuid4()) for _ in range(len(docs))]
        vector_store.add_documents(documents=docs, ids=uuids)

    return vector_store


def extract_style_instructions(profile_text):
    """Extract actionable style instructions from the profile"""
    instructions = []

    # Key characteristics from the profile
    if "Average message length: 29.9 words" in profile_text:
        instructions.append("Keep responses SHORT - around 30 words or less")
    if "Average sentence length: 7.5 words" in profile_text:
        instructions.append("Use SHORT sentences - about 7-8 words each")
    if "Uses dashes: Yes" in profile_text:
        instructions.append("Use dashes (-) frequently in your writing")
    if "🤔" in profile_text or "🎉" in profile_text:
        instructions.append("Use emojis like 🤔 and 🎉 naturally in responses")
    if "Formality score: 0.59" in profile_text:
        instructions.append("Write in a CASUAL, conversational tone - not formal")
    if "'hi'" in profile_text or "'hey'" in profile_text:
        instructions.append("Use casual greetings like 'hi' or 'hey' when appropriate")

    # Add general style requirements
    instructions.append("Use simple, everyday words")
    instructions.append("Start with casual greetings when appropriate (hi, hey)")

    return instructions


def format_style_instructions(instructions):
    """Format style instructions as bullet points and numbered list"""
    numbered_list = "\n".join(f"{i+1}. {inst}" for i, inst in enumerate(instructions))
    return numbered_list


def generate_style_examples(profile_text, llm):
    """Generate style examples using LLM based on the style profile

    Args:
        profile_text: The style profile text
        llm: LLM instance for generating examples
    """
    from langchain_core.messages import SystemMessage, HumanMessage

    prompt = f"""Based on this style profile, generate EXACTLY 3 short example responses that demonstrate the style:

{profile_text}

CRITICAL: Generate ONLY the 3 examples in this EXACT format (no other text, no explanations):

- Short, casual response: "example text here"
- Use dashes: "example text here"
- Keep it brief: "example text here"

Each example must:
- Be SHORT (around 20-30 words max, 2-3 sentences)
- Start with casual greeting like "hi" or "hey" when appropriate
- Use dashes (-) frequently to connect thoughts
- Include emojis like 🤔 or 🎉 naturally
- Be casual and conversational - like texting a friend
- Use short sentences (7-8 words each)
- NO formal language, NO "Dear", NO "Best regards"
- NO numbered lists, NO long paragraphs

Generate ONLY the 3 lines above, nothing else:"""

    messages = [
        SystemMessage(
            content="You are a helpful assistant. Generate ONLY the requested format, no additional text."
        ),
        HumanMessage(content=prompt),
    ]
    result = llm.invoke(messages)
    if hasattr(result, "content") and result.content:
        # The LLM should return exactly the format we requested
        content = result.content.strip()
        # Split by lines and filter for lines starting with -
        lines = [
            line.strip() for line in content.split("\n") if line.strip().startswith("-")
        ]
        if len(lines) >= 3:
            return "\n".join(lines[:3])
        elif len(lines) > 0:
            # If we got some lines, return them
            return "\n".join(lines)
        else:
            # If no lines found, return the content as-is (might be in a different format)
            return content

    raise ValueError("LLM returned empty or no content")


vector_store = create_vector_store()
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

model_id = "HuggingFaceH4/zephyr-7b-beta"
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Ensure tokenizer has a pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
    load_in_4bit=True,  # memory-efficient quantization
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,  # Reduced to encourage shorter responses
    temperature=0.7,  # Increased for more creative/casual responses
    top_p=0.95,
    do_sample=True,  # Enable sampling for more diverse outputs
    return_full_text=False,  # Don't return the input prompt in the output
    repetition_penalty=1.2,  # Critical: prevents repetitive outputs
    pad_token_id=tokenizer.eos_token_id,  # Set pad token to prevent issues
    eos_token_id=tokenizer.eos_token_id,  # Explicitly set EOS token
)

# Use ChatHuggingFace directly - it handles chat templates automatically
hf_pipeline = HuggingFacePipeline(pipeline=pipe)
llm = ChatHuggingFace(
    llm=hf_pipeline,
    model_id=model_id,
    tokenizer=tokenizer,
    max_new_tokens=200,  # Reduced to encourage shorter responses
)

# Generate style examples using LLM
print("Generating style examples using LLM...")
style_examples = generate_style_examples(style_profile, llm)
style_instructions_list = extract_style_instructions(style_profile)
style_instructions_numbered = format_style_instructions(style_instructions_list)
print("Style examples:")
print(style_examples)
print("Style instructions numbered:")
print(style_instructions_numbered)

# Create prompt with style profile
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""You are a helpful assistant that answers questions based on the provided context.

CRITICAL INSTRUCTION: Your response MUST match the communication style specified below. Style matching is MANDATORY and takes priority. Before writing your answer, carefully study the style examples and requirements.

STYLE REQUIREMENTS - YOU MUST FOLLOW THESE:
{style_instructions_numbered}

STYLE EXAMPLES - YOUR RESPONSE MUST MATCH THIS EXACT STYLE:
{style_examples}

BEFORE YOU RESPOND:
1. Read the style examples above carefully
2. Count the words in the examples - match that length
3. Notice the sentence structure - use the same pattern
4. Observe the tone and formality level - replicate it exactly
5. Check punctuation usage (dashes, emojis, etc.) - use them the same way
6. Review common phrases and greetings - incorporate them naturally

YOUR RESPONSE WILL BE WRONG IF:
- It doesn't match the length of the style examples
- It uses a different tone (too formal or too casual)
- It doesn't follow the punctuation patterns shown
- It doesn't use the same sentence structure
- It doesn't incorporate the style characteristics from the examples
- It does not use emoji of dashes and example text shows it should be used


Full style profile for detailed reference:
{style_profile}""",
        ),
        (
            "user",
            """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {input}

IMPORTANT: Before writing your answer, review the style examples and requirements in the system message. Your answer MUST match that exact style - check length, tone, punctuation, and sentence structure.

Answer:""",
        ),
    ]
)

combine_docs_chain = create_stuff_documents_chain(llm, prompt)
limited_retriever = vector_store.as_retriever(search_kwargs={"k": 2})
rag_chain = create_retrieval_chain(limited_retriever, combine_docs_chain)


@tool
def retrieve_and_answer(query: str) -> str:
    """Retrieve information from knowledge base and answer questions using RAG.

    Use this tool when the user asks questions that might be answered from the knowledge base.
    The tool will search the knowledge base and provide an answer in the appropriate style.

    Args:
        query: The user's question or query

    Returns:
        An answer based on the retrieved context, formatted in the appropriate style
    """
    print(f"[TOOL CALL] retrieve_and_answer")
    print(f"   query={query}")
    try:
        result = rag_chain.invoke({"input": query})
        answer = result.get("answer", "") if isinstance(result, dict) else ""
        final_answer = (
            answer
            if answer
            else "I couldn't find a relevant answer in the knowledge base."
        )
        print(
            f"   → Result: {final_answer[:200]}..."
            if len(final_answer) > 200
            else f"   → Result: {final_answer}"
        )
        return final_answer
    except Exception as e:
        error_msg = f"Error retrieving answer: {str(e)}"
        print(f"   → Error: {error_msg}")
        return error_msg


# Test question (only run if this file is executed directly)
if __name__ == "__main__":
    query = "My daughter won't listen to me about anything. Just ell me what to do?"

    try:
        result = rag_chain.invoke({"input": query})
        answer = result.get("answer", "") if isinstance(result, dict) else ""
        print("--------------------------------\n\n")
        print("RAG AGENT RESPONSE:")
        print("--------------------------------\n\n")
        print(answer)
        print("--------------------------------\n\n")

    except Exception as e:
        print(f"\nERROR during RAG chain execution: {e}")
        import traceback

        traceback.print_exc()
