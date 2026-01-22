import os 
import google.generativeai as genai
from pdfextractor import text_extractor
import streamlit as st

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter



# Lets create the main page 
st.title(':orange[CHATBOT:] :blue[AI Assisted Chatbot using RAG]')

tips='''
Follow the steps to use the application
* Upload Your PDF Document in Sidebar.
* Write a Query and Start the chat.'''

st.text(tips)

#Lets configure the models

# Step1 : Configure the Models
# First lets configure the Models
#LLM Model

gemini_key = os.getenv('Google_API_Key2')
genai.configure(api_key=gemini_key)
model = genai.GenerativeModel('gemini-2.5-flash-lite')

# Configure Embedding Model
embedding_model  = HuggingFaceBgeEmbeddings(model_name='all-MiniLM-L6-v2')

# Lets create sidebar 
st.sidebar.title(':green[UPLOAD YOUR FILE]')
st.sidebar.subheader(':red[Upload PDF File Only]')
pdf_file = st.sidebar.file_uploader('Upload here',type=['pdf'])


if 'history' not in st.session_state:
    st.session_state.history = []

if pdf_file:
    st.sidebar.success('File Uploaded Successfully')

    file_text = text_extractor(pdf_file)

    # step 1 :Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunks = splitter.split_text(file_text)

    # Step 2: create the vector Data Base (FAISS)
    vector_store = FAISS.from_texts(chunks,embedding_model)
    retriever = vector_store.as_retriever(search_kwargs={'k':3})

    def generate_content(query):
        #step3: Retrieval (R)
        retrived_docs = retriever.invoke(query)
        context = '\n'.join([d.page_content for d in retrived_docs])

        chat_history_str = ""
        for chat in st.session_state.history:
            role = "User" if chat['role'] == 'user' else "model"
            chat_history_str += f"{role}: {chat['text']}\n"

        augmented_prompt = f''' <Role> You are an helpful Assitant(model) Using RAG

        <Goal> Answer the Question asked by the user. Here is the <question> {query} <question> 
        <Instructions>You must consider the Context provided and the Conversation History.
        
        <Conversation History>
        {chat_history_str}
        <Conversation History>
        <context>here are the douments retreived from the vector Database to support the answer which you have to generate : {context}'''

        response = model.generate_content(augmented_prompt)
        return response.text 

    
    # Display the history 
    for msg in  st.session_state.history:
        if msg['role']=='user':
            st.info(f':green[User:] :blue[{msg['text']}]')
        else:
            st.warning(f':orange[Chatbot:   :blue[{msg['text']}]]')
    # Input from the user using streamlit form
    with st.form('Chatbot Form',clear_on_submit= True):
        user_query = st.text_area('Ask Anyhting')
        send = st.form_submit_button('Send')

    # Start the Conversation and append output and query in history
    if user_query and send:
        st.session_state.history.append({'role':'user','text':user_query})
        st.session_state.history.append({'role':'model','text':generate_content(user_query)})
        st.rerun()