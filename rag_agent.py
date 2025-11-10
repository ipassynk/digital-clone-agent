from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, json
from langchain.tools import tool
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

with open("style_summary.txt", "r") as f:
    style_profile = f.read()

# Extract key style characteristics for concise instructions
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
    bullet_points = "\n".join(f"  • {inst}" for inst in instructions)
    numbered_list = "\n".join(f"{i+1}. {inst}" for i, inst in enumerate(instructions))
    return bullet_points, numbered_list

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
- Be short (around 30 words max)
- Use dashes frequently
- Include emojis like 🤔 or 🎉
- Be casual and conversational
- Match the style characteristics described

Generate ONLY the 3 lines above, nothing else:"""
    
    messages = [
        SystemMessage(content="You are a helpful assistant. Generate ONLY the requested format, no additional text."),
        HumanMessage(content=prompt)
    ]
    result = llm.invoke(messages)
    if hasattr(result, 'content') and result.content:
        # The LLM should return exactly the format we requested
        content = result.content.strip()
        # Split by lines and filter for lines starting with -
        lines = [line.strip() for line in content.split('\n') if line.strip().startswith('-')]
        if len(lines) >= 3:
            return "\n".join(lines[:3])
        elif len(lines) > 0:
            # If we got some lines, return them
            return "\n".join(lines)
        else:
            # If no lines found, return the content as-is (might be in a different format)
            return content
    
    raise ValueError("LLM returned empty or no content")

style_instructions_list = extract_style_instructions(style_profile)
style_instructions_bullets, style_instructions_numbered = format_style_instructions(style_instructions_list)

# Style examples will be generated using LLM after it's created
style_examples = None

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = InMemoryVectorStore(embeddings)

# Load the kids issues article from text file
loader = TextLoader("kids_issues_article.txt")
docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)
print(f"Split blog post into {len(all_splits)} sub-documents.")
document_ids = vector_store.add_documents(documents=all_splits)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})  # Limit to 2 documents to reduce token count

model_id = "HuggingFaceH4/zephyr-7b-beta"

tokenizer = AutoTokenizer.from_pretrained(model_id)
# Ensure tokenizer has a pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Zephyr has excellent built-in chat template support with system message handling
# No need for custom template - the built-in one works perfectly

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
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
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
    max_new_tokens=512,
)

# Generate style examples using LLM
print("Generating style examples using LLM...")
style_examples = generate_style_examples(style_profile, llm)
print(f"Generated examples:\n{style_examples}\n")

# Create prompt with style profile
prompt = ChatPromptTemplate.from_messages([
    ("system", f"""You are a helpful assistant that answers questions based on the provided context.

CRITICAL STYLE REQUIREMENTS - You MUST follow these exactly:

{style_instructions_bullets}

STYLE EXAMPLES:
{style_examples}

When answering questions:
{style_instructions_numbered}

DO NOT write formal letters, long paragraphs, or formal language. Write like you're texting a friend - short, casual, with dashes and emojis.

Full style profile for reference:
{style_profile}"""),
    ("user", """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {input}

Answer:""")
])

# Create the combine documents chain
combine_docs_chain = create_stuff_documents_chain(llm, prompt)

# Limit the number of documents retrieved to reduce token count
limited_retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# Create the retrieval chain
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
        final_answer = answer if answer else "I couldn't find a relevant answer in the knowledge base."
        print(f"   → Result: {final_answer[:200]}..." if len(final_answer) > 200 else f"   → Result: {final_answer}")
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
        print(answer)

        
    except Exception as e:
        print(f"\nERROR during RAG chain execution: {e}")
        import traceback
        traceback.print_exc()

