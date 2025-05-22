from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import streamlit as st

# Load environment variables (e.g., GROQ_API_KEY)
load_dotenv()

# Load PDF document
loader = PyPDFLoader(r"E:\KodEent_Chatbot\Shakil Rana.pdf")
docs = loader.load()

# Split text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
split_docs = splitter.split_documents(docs)

# Embedding and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(split_docs, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# Define prompt template
prompt = ChatPromptTemplate.from_template("""
You have to act like Shakil Rana. Your bio will be given in the context.
People will ask questions to you and you must answer them based on the provided context.
Please provide the answer most accurately and concisely.

<context>
{context}
</context>

Question: {input}
Answer:
""")

# Load Groq LLM
llm = ChatGroq(model="deepseek-r1-distill-llama-70b")

# Create document QA chain
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Streamlit UI
st.set_page_config(page_title="Shakil Rana Chatbot", page_icon="🤖")
st.header("🤖 Shakil Rana Chatbot")
st.write("Ask anything based on Shakil Rana's bio.")

user_input = st.text_input("Enter your question:")

if st.button("Submit"):
    if user_input.strip():
        response = retrieval_chain.invoke({'input': user_input})
        st.write(response['answer'].split('</think>')[-1])
    else:
        st.warning("Please enter a question.")
