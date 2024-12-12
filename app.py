# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st 
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_openai import ChatOpenAI
from uuid import uuid4
import pandas as pd 
import PyPDF2
import uuid
import os
from pyunpack import Archive
from utils.functions import create_conversational_rag_chain


os.environ["OPENAI_API_KEY"] = "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"
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

names = []
system_prompt_dirs = []
vdb_dirs = []
BAAI = 'BAAI/bge-base-en-v1.5'
L6 = 'sentence-transformers/all-MiniLM-L6-v2'
names = []
system_prompt_dirs = []
vdb_dirs = []

for dir in os.listdir():
     if os.path.isdir(dir) and "." not in dir and dir != "utils": 
        names.append(dir)
        system_prompt_dirs.append(f"{dir}/system_prompt.txt")
        vdb_dirs.append(f"{dir}/Vdb/Vdb/"
        

configirations = {'name': names,
                  'embeddings_name':[BAAI,BAAI,BAAI,BAAI,BAAI,BAAI,BAAI,BAAI],
                  'vdb_dir':vdb_dirs,
                  'sys_prompt_dir':system_prompt_dirs}
df = pd.DataFrame(configirations)


def bot_func(rag_chain,user_input, session_id):
    for chunk in rag_chain.stream({"input": user_input}, config={"configurable": {"session_id": session_id}}):
        if answer_chunk := chunk.get("answer"):
            yield answer_chunk

bots = {}
for index, row in df.iterrows():
    embeddings = row["embeddings_name"]
    vdb_dir = row["vdb_dir"]
    sys_prompt_dir = row["sys_prompt_dir"]
    rag_chain = create_conversational_rag_chain(sys_prompt_dir,vdb_dir,llm,embeddings)
    bots[row["name"]] = rag_chain



def extract_pdf_text(file_object):
    reader = PyPDF2.PdfReader(file_object)
    text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text()
    return text


st.title("business schools 🤖")



if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

st.write(f"Your session ID: {st.session_state.session_id}")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "selected_bot" not in st.session_state:
    st.session_state.selected_bot = None

selected_bot = st.selectbox("Choose a bot to interact with:", list(bots.keys()))


if selected_bot != st.session_state.selected_bot:
    st.session_state.chat_history = []  
    st.session_state.selected_bot = selected_bot

uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")

user_input = st.chat_input("Ask whatever you want")
user_input_2 = user_input
if uploaded_file is not None:
    text = extract_pdf_text(uploaded_file)
    user_input += text

for sender, message in st.session_state.chat_history:
    with st.chat_message("user" if sender == "You" else "assistant"):
        st.markdown(message)
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response_palceholder = st.empty()
        full_response = "" 
        for response in bot_func(bots[selected_bot],user_input,session_id= st.session_state.session_id):
            full_response += response
            response_palceholder.markdown(full_response) 



    st.session_state.chat_history.append(("You", user_input_2))

    st.session_state.chat_history.append((selected_bot, full_response))
