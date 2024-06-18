from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
import os
import pdfplumber

from langchain_community.vectorstores import FAISS
from langchain_voyageai import VoyageAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_core.messages import SystemMessage
import os
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,)

groq_api_key = os.environ.get('GROQ_KEY')
voyage_api_key = os.environ.get('VOYAGE_KEY')

groq_api_key = st.secrets["GROQ_KEY"]
voyage_api_key = st.secrets["VOYAGE_KEY"]

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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
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
    print(relevant_excerpts)
    return relevant_excerpts

memory=ConversationBufferMemory(memory_key="history", return_messages=True)

def chat_completion(client, user_question, relevant_excerpts, additional_context):
    system_message = '''
    Give simple and understandable answers defaulting to the most relevant information from the PDF files. If there is no relevent content, say "No relevant content found".
    '''
    # Add the additional context to the system prompt if it's not empty
    if additional_context != '':
        system_prompt += '''\n
        The user has provided this additional context:
        {additional_context}
        '''.format(additional_context=additional_context)

        # Generate a response to the user's question using the pre-trained model
    
    human_message = user_question


    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_message),  # The persistent system prompt
            MessagesPlaceholder(variable_name="history"),  # The conversation history
            HumanMessagePromptTemplate.from_template("{input}"),  # The user's current input
        ]
    )

    # Create the conversation chain
    chain = ConversationChain(
        memory=memory,
        llm=client,
        verbose = False,
        prompt=prompt,
    )

    # Predict the answer
    answer = chain.predict(input=human_message)
    
    # Save the context
    memory.save_context({"input": human_message}, {"output": answer})
    return answer

def final():
    # Display the title and introduction of the application
    st.title("Chat with your PDF files")
    st.markdown('---')

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
    st.sidebar.markdown("""---""")
    additional_context = st.sidebar.text_input('Enter additional summarization context for the LLM here (i.e. write it in simple):')
    
    # Initialize the Groq client
    model = st.sidebar.selectbox(
        'Choose a model',
        ['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it'])
    client = ChatGroq(temperature=0.7,model=model, groq_api_key=groq_api_key)


    # Initialize chat history if not already done
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if 'chat_completion' not in st.session_state:
        st.session_state.chat_completion = chat_completion

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        avatar = "🤖" if message["role"] == "assistant" else "👨‍💻"
        with st.chat_message(message["role"], avatar=avatar):
            st.write(message["content"])
    
    user_question = st.chat_input("Ask a question about the PDF files")
    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user", avatar="👨‍💻"):
                st.markdown(user_question)
        relevant_excerpts = get_relevant_excerpts(user_question)
        response = st.session_state.chat_completion(client, user_question, relevant_excerpts, additional_context)

        # Add the responses to the chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant", avatar="🤖"):
            st.write(response)


final()