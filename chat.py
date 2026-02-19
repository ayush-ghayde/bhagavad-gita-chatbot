import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

# Streamlit Title
st.title("Bhagavad Gita Spiritual Chatbot 🕉️")

# Session Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Previous Messages
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])


# -------------------------------
# Load FAISS Vector Database
# -------------------------------
@st.cache_resource
def load_my_vectorstore():
    DB_FAISS_PATH = "vectorstore/db_faiss"

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return db


# -------------------------------
# User Input
# -------------------------------
user_prompt = st.chat_input("Ask your Bhagavad Gita question here...")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    try:
        # Load Vectorstore
        db = load_my_vectorstore()

        # Load Groq LLM
        llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant"
        )

        # -------------------------------
        # Updated Bhagavad Gita Prompt
        # -------------------------------
        template = """
You are a Bhagavad Gita expert and spiritual guide.

Your task is to answer the user's question ONLY using teachings from the Bhagavad Gita.

====================================================
RULES
====================================================

1. Answer must be strictly based on Bhagavad Gita wisdom.
2. The answer must be in English.
3. You MUST mention at least one relevant shloka reference inside the answer.
   Format: (*Bhagavad Gita, Chapter X, Verse Y*)

4. After the answer, you MUST provide the exact Sanskrit shloka text in proper transliteration
   with diacritical marks (example: śrī kṛṣṇa uvāca...).

5. Sanskrit words must include English meaning in parentheses.
6. Keep the answer clear, practical, and connected to real life.

====================================================
OUTPUT FORMAT (Strict)
====================================================

ANSWER: [Write the full answer here in continuous text with inline shloka reference]

SHLOKA TEXT: *[Write the Sanskrit shloka in Roman transliteration with diacritics, like: asaṁśayaṁ mahābāho...]*

SHLOKA REF: [Only list the shloka reference, semicolon-separated]

====================================================
Context from Database:
{context}

User Question: {question}

Now generate the response strictly in the required format.
"""

        QA_PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        # -------------------------------
        # RetrievalQA Chain
        # -------------------------------
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": QA_PROMPT}
        )

        # Run Chain
        response = chain.run(user_prompt)

        # Display Response
        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

    except Exception as e:
        st.error(f"Error: {str(e)}")
