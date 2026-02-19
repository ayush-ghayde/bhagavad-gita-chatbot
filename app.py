import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

st.title('Personal  Medical Chatbot! 🏥')

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Optimized Database Loading
@st.cache_resource
def load_my_vectorstore():
    DB_FAISS_PATH = "vectorstore/db_faiss"
    embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    

    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

prompt = st.chat_input('Ask your medical question here...')

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role':'user', 'content': prompt})
    
    try:
        
        db = load_my_vectorstore() 
        
        llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-8b-instant")

        template = """Answer the question based ONLY on the context. 
        Context: {context}
        Question: {question}

        Answer:
        ⚠️ DISCLAIMER: This information is provided for educational purposes only. Please consult a certified doctor or medical professional before starting any treatment or medication."""
        
        QA_PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

        
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=db.as_retriever(search_kwargs={'k': 3}),
            chain_type_kwargs={"prompt": QA_PROMPT}
        )
       
        response = chain.run(prompt)
        st.chat_message('assistant').markdown(response)
        st.session_state.messages.append({'role':'assistant', 'content':response})

    except Exception as e:
        st.error(f"Error: {str(e)}")