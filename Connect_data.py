import os
from dotenv import load_dotenv, find_dotenv
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# Load environment
load_dotenv(find_dotenv())
HF_TOKEN = os.getenv("HF_TOKEN")

#  Model selection
HUGGINGFACE_REPO_ID = "microsoft/Phi-4-mini-instruct"

def load_llm_chat():
    print(f" Step 1: Connecting to Chat Interface ({HUGGINGFACE_REPO_ID})...")
    
    llm = HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        huggingfacehub_api_token=HF_TOKEN,
        task="conversational", 
        temperature=0.5,
    )
    # Wrapping it for Chat compatibility
    return ChatHuggingFace(llm=llm)

# Step 2: Database Loading
print(" Step 2: Loading FAISS Database...")
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Step 3: Prompt & Chain

prompt = f"""
You are a subject matter expert in Bhagavad Gita and Indian spiritual philosophy.
Your task is to generate 5 high-quality Questions and Answers about **{topic}**.

Generate relevant questions necessary to comprehensively cover this topic. Aim for 5 unique questions, but prioritize depth and non-repetition.

====================================================
                CORE REQUIREMENTS
====================================================
1. **Source Material**: Answers must be derived STRICTLY from:
   - Bhagavad Gita (primary source)
   - Supporting references may include Upanishads or Mahabharata only if needed.

   Each answer MUST include at least one relevant shloka reference such as:
   - Bhagavad Gita, Chapter X, Verse Y

2. **Language**:
   - The output must be in English.
   - For all Sanskrit words used, include English meanings in parentheses.

3. **Shloka References**:
   - You MUST include specific shloka citations inside the running text.
   - References must be italicized.
   - Format: (*Bhagavad Gita, Chapter X, Verse Y*)

4. **Life Application Focus**:
   - Answers should connect Gita wisdom to real-life situations such as stress, duty, fear, success, failure, relationships, and self-growth.

====================================================
            ANSWER STRUCTURE & HEADERS
====================================================
Answer Format Instructions (Continuous Text Form)

Please provide answers in a single continuous block of text without paragraph breaks, line breaks, headings, bullet points, numbering, or separate lines.

Guidelines for Answers:
Begin with a brief introductory explanation and continue smoothly into the explanation.
Explain the spiritual teaching clearly and connect it to modern life challenges.
Mention relevant Sanskrit concepts with translations naturally.
Conclude within the same continuous flow without adding a separate concluding section.
Do not include diagrams, flowcharts, graphs, or formatting elements.

====================================================
              STRICT OUTPUT FORMAT
====================================================
The output MUST be valid Markdown.
For EACH Q&A pair, use exactly the format below. Do not deviate.

Q1: [Question text]

A1: [Full Answer in continuous text with inline shloka references]

SHLOKAS1: [List only the shlokas cited in A1, semicolon-separated]

TAGS1: [4-5 keywords, comma-separated]

Q2: [Question text]

A2: [Full Answer in continuous text with inline shloka references]

SHLOKAS2: [List only the shlokas cited in A2, semicolon-separated]

TAGS2: [4-5 keywords, comma-separated]

(Continue strictly in this format for all generated questions)

====================================================
              QUALITY CONSTRAINTS
====================================================
- All questions must be unique; do not rephrase the same question.
- If {topic} is fully covered by fewer than 5 questions, do not force duplicates.
- Ensure all content remains strictly within Bhagavad Gita teachings.
- The SHLOKAS section must ONLY contain references mentioned in that specific answer.
- Shlokas must include chapter and verse numbers correctly.
"""


prompt = PromptTemplate(
    template=prompt, 
    input_variables=["question"]
)




qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm_chat(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': prompt}
)

# Step 4: Run
user_query = input("\n💬 ASK QUERY HERE: ")
print("🔍 Searching context and generating answer...")

try:
    response = qa_chain.invoke({'query': user_query})
    print("\n RESULT: ", response["result"])
    print("\n SOURCE DOCUMENTS: ", [doc.metadata for doc in response["source_documents"]])
except Exception as e:
    print(f" Error logic update needed: {e}")