from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
import os
from groq import Groq
import pdfplumber

from langchain_community.vectorstores import FAISS
from langchain_voyageai import VoyageAIEmbeddings

groq_api_key = os.environ.get('GROQ_KEY')
voyage_api_key = os.environ.get('VOYAGE_KEY')
embedd_model = VoyageAIEmbeddings(voyage_api_key=voyage_api_key, model="voyage-law-2")


# Read files from stremlit sidebar
def get_pdf_text_from_sidebar(pdf_files):
    text = ''
    for file in pdf_files:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
    return text
# Split into chunks
def get_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(raw_text)
    return chunks


# Create vector store 
def get_vector_store(chunk_texts):
    try:
        documents = chunk_texts
        vector_store = FAISS.from_texts(texts=documents, embedding=embedd_model)
        # Save the vector store locally
        vector_store.save_local("faiss_index")
        return {"message": "The vector store has been created successfully."}
    except Exception as e:
        return {"message": f"An error occurred: {e}."}


def get_relevant_excerpts(user_question):
    vectors = FAISS.load_local("faiss_index", embedd_model,allow_dangerous_deserialization=True)
    docs = vectors.similarity_search(user_question)
    relevant_excerpts = '\n\n------------------------------------------------------\n\n'.join([doc.page_content for doc in docs[:3]])
    return relevant_excerpts


def chat_completion(client, model, user_question, relevant_excerpts, additional_context):
    
    system_prompt = '''
    You are a statistician.
    '''
    # Add the additional context to the system prompt if it's not empty
    if additional_context != '':
        system_prompt += '''\n
        The user has provided this additional context:
        {additional_context}
        '''.format(additional_context=additional_context)

        # Generate a response to the user's question using the pre-trained model
    chat_completion = client.chat.completions.create(
        messages = [
            {
                "role": "system",
                "content":  system_prompt
            },
            {
                "role": "user",
                "content": "User Question: " + user_question + "\n\nRelevant Speech Exerpt(s):\n\n" + relevant_excerpts,
            }
        ],
        model = model
    )
    
    # Extract the response from the chat completion
    response = chat_completion.choices[0].message.content
    return response

def final():
    # Initialize the Groq client
    client = Groq(
        api_key=groq_api_key
    )
    spacer, col = st.columns([5, 1])  
    with col:  
        st.markdown('###### [Contact Me](https://github.com/Samitha10)')
        st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Samitha10/RAG-Chatbot)")
    # Display the title and introduction of the application
    st.title("Chat with your PDF files")
    multiline_text = """Welcome! Ask questions about from the PDF files."""

    st.markdown(multiline_text, unsafe_allow_html=True)

    # Add customization options to the sidebar
    st.sidebar.title('Customization')
    pdf_docs = st.sidebar.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)

    if st.sidebar.button("Submit & Process"):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text_from_sidebar(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("Your Chatbot is ready.")
    additional_context = st.sidebar.text_input('Enter additional summarization context for the LLM here (i.e. write it in simple):')
    model = st.sidebar.selectbox(
        'Choose a model',
        ['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it'])
    # Get the user's question


    user_question = st.text_input("Ask a question")

    # conversational_memory_length = st.sidebar.slider('Conversational memory length:', 1, 10, value = 5)
    # memory=ConversationBufferWindowMemory(k=conversational_memory_length)

    if user_question:
        relevant_excerpts = get_relevant_excerpts(user_question)
        response = chat_completion(client, model, user_question, relevant_excerpts, additional_context)
        st.write(response)
        message = {'human': user_question, 'AI': response}

        # Update session state variable
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = [message]  # Initialize chat history
        else:
            st.session_state.chat_history.append(message)  # Append new message to chat history

        # Display conversation history
        st.subheader("Conversation History")
        for message in st.session_state.chat_history:
            st.write(message)


final()