# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import pickle
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import StreamlitChatMessageHistory


msgs = StreamlitChatMessageHistory(key="special_app_key")




def read_data(filepath: str):
    """
    Function to read the data from pkl file 
    
    Parameters:
        filepath(str): The directory to the pkl file
    
    Returns:
        the document objects stored in a variable
    """
    
    with open(filepath,'rb')as f:
        data = pickle.load(f)
    return data    

def read_db(filepath: str, embeddings_name):
    """
    Function to read the vector database and assign at is retreiver
    
    Parameters:
        vdb_dir(str): the directory where the vector database is located
        embeddings(str): the embeddings name 
    
    Returns:
        the retreiver     
    """
    embeddings = HuggingFaceEmbeddings(model_name = embeddings_name)
    vectordb = Chroma(persist_directory=filepath, embedding_function=embeddings)
    retreiver = vectordb.as_retriever()
    
    # take it as ret
    return retreiver


def read_system_prompt(filepath: str):
    """
    Function to read the system prompt
    
    Parameters:
        sys_prompt_dir(str): the directory where the system prompt is located
    
    Returns:
        The system prompt stored in variable
    """
    with open(filepath, 'r') as file:
        prompt_content = file.read()

    context = "{context}"

    system_prompt = f'("""\n{prompt_content.strip()}\n"""\n"{context}")'

    return system_prompt



def create_conversational_rag_chain(sys_prompt_dir,vdb_dir,llm,embeddings_name):
    retriever = read_db(vdb_dir,embeddings_name)
    
    
    contextualize_q_system_prompt = (
    """Given a chat history and the latest user question
    which might reference context in the chat history,
    formulate a response which can be understood and clear
    without the chat history. Do NOT answer the question,
    """
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # add chat history
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    
    )
    sys_prompt = read_system_prompt(sys_prompt_dir)

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", sys_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)   


    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        store = {}
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]
    
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
        output_messages_key="answer",
        max_tokens_limit=500,
        top_n=5
    )
    return conversational_rag_chain