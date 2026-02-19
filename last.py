import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import base64

# Load Environment Variables
load_dotenv()

# ----------------------------------------
# PAGE CONFIG
# ----------------------------------------
st.set_page_config(
    page_title="Bhagavad Gita Chatbot",
    page_icon="🕉️",
    layout="wide"
)

# ----------------------------------------
# BACKGROUND + PREMIUM UI THEME
# ----------------------------------------
def add_bg_from_local(image_file):
    with open(image_file, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()

    st.markdown(
        f"""
        <style>

        /* FULL BACKGROUND */
        .stApp {{
            background: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        /* REMOVE STREAMLIT HEADER */
        header {{
            visibility: hidden;
        }}

        /* CHAT CONTAINER */
        .chat-container {{
            max-width: 900px;
            margin: auto;
        }}

        /* USER MESSAGE BUBBLE */
        .user-box {{
            background: rgba(255, 255, 255, 0.80);
            color: #1a1a1a;
            padding: 14px 18px;
            border-radius: 18px;
            margin: 14px 0px;
            font-size: 17px;
            font-family: Georgia, serif;
            border: 1px solid #c2a46d;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.25);
            width: fit-content;
        }}

        /* BOT MESSAGE BUBBLE */
        .bot-box {{
            background: rgba(255, 248, 220, 0.90);
            color: #2b1b0f;
            padding: 16px 20px;
            border-radius: 18px;
            margin: 14px 0px;
            font-size: 17px;
            font-family: Georgia, serif;
            border: 1px solid #c2a46d;
            box-shadow: 2px 2px 12px rgba(0,0,0,0.30);
            width: fit-content;
        }}

        /*  DARK INPUT BAR */
        textarea {{
            border-radius: 18px !important;
            font-size: 16px !important;
            padding: 12px !important;

            background-color: rgba(20, 20, 20, 0.92) !important;
            color: white !important;

            border: 1px solid #c2a46d !important;
        }}

        /* Placeholder Text Light */
        textarea::placeholder {{
            color: rgba(200, 200, 200, 0.8) !important;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

# ----------------------------------------
# ADD YOUR BACKGROUND IMAGE
# ----------------------------------------
add_bg_from_local("Gemini_Generated_Image_q6aeniq6aeniq6ae.png")

# ----------------------------------------
# PREMIUM TITLE
# ----------------------------------------
st.markdown(
    """
    <h1 style='text-align:center;
    font-family: Georgia, serif;
    color: #3b1f0f;
    font-size: 65px;
    font-weight: bold;
    text-shadow: 2px 2px 8px #f5deb3;
    margin-bottom: 5px;'>
    Bhagavad Gita Spiritual Chatbot 🕉️
    </h1>

    <p style='text-align:center;
    font-family: Georgia;
    font-size: 20px;
    color: #2b1b0f;
    text-shadow: 1px 1px 4px white;'>
    Seek guidance from Krishna’s divine wisdom
    </p>

    <hr style="border:1px solid #c2a46d;">
    """,
    unsafe_allow_html=True
)

# ----------------------------------------
# SESSION CHAT HISTORY
# ----------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------------------
# LOAD FAISS VECTORSTORE
# ----------------------------------------
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

# ----------------------------------------
# DISPLAY CHAT MESSAGES
# ----------------------------------------
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(
            f"<div class='user-box'><b>You:</b> {message['content']}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='bot-box'><b>Krishna:</b><br>{message['content']}</div>",
            unsafe_allow_html=True
        )

st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------
# USER INPUT
# ----------------------------------------
user_prompt = st.chat_input("Ask your Bhagavad Gita question here...")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    try:
        db = load_my_vectorstore()

        llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant"
        )

        template = """
You are a Bhagavad Gita expert and spiritual guide.

Answer ONLY using Bhagavad Gita teachings.

IMPORTANT RULES:
- The ANSWER must be strictly 2 to 3 complete sentences (minimum 2 lines).
- Do NOT write the answer in a single short line.
- Must mention the MOST relevant shloka for the question
- Must mention shloka reference (Chapter:Verse)
- Provide Sanskrit transliteration with diacritics
- Keep practical and clear

FORMAT:

ANSWER: (2–3 lines)

SHLOKA TEXT (Most Relevant): ...

TRANSLITERATION: ...

SHLOKA REF: ...

Context:
{context}

Question:
{question}
"""


        QA_PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": QA_PROMPT}
        )

        response = chain.run(user_prompt)

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

        st.rerun()

    except Exception as e:
        st.error(f"Error: {str(e)}")
