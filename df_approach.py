import streamlit as st 
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_openai import ChatOpenAI
from utils.functions import create_conversational_rag_chain
from uuid import uuid4
import pandas as pd 
import os 

llm = ChatOpenAI(model="gpt-4o-mini")

configirations = {'name': ["ESAM","FFOLLOZZ"],
                  'embeddings_name':["BAAI/bge-base-en-v1.5","sentence-transformers/all-MiniLM-L6-v2"],
                  'vdb_dir':['esam','ffollozz'],
                  'sys_prompt_dir':['esam/ESAM sys propmt.txt','ffollozz/ffollozz_system_prompt.txt']}
df = pd.DataFrame(configirations)
def bot_func(rag_chain,user_input, session_id):
    response = rag_chain.invoke({"input": user_input}, config={"configurable": {"session_id": session_id}})["answer"]
    return str(response)
bots = {}
for index, row in df.iterrows():
    embeddings = row["embeddings_name"]
    vdb_dir = row["vdb_dir"]
    sys_prompt_dir = row["sys_prompt_dir"]
    rag_chain = create_conversational_rag_chain(sys_prompt_dir,vdb_dir,llm,embeddings)
    bots[row["name"]] = rag_chain
    
st.title("IGS chatbots")


session_id = str(uuid4())

selected_bot = st.selectbox("Choose a bot to interact with:", list(bots.keys()))


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You", placeholder="ask whatever you want")



if st.button("Send"):
    if user_input:
        response = bot_func(bots[selected_bot],user_input,session_id= session_id)

        st.session_state.chat_history.append(("You", user_input))

        st.session_state.chat_history.append((selected_bot, response))

for sender, message in st.session_state.chat_history:
    if sender == "You" or sender == selected_bot:
        st.markdown(f"{sender}: {message}")
