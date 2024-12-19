import os
import pickle
import requests
from bs4 import BeautifulSoup
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# Retrieve API keys from .env
groq_api_key = os.getenv("GROQ_API_KEY")

# Streamlit UI
st.title("Task-2: Chat with Website Using RAG Pipeline")
st.sidebar.title("Website Content Chatbot")

# Input URLs for crawling
website_urls = st.sidebar.text_area("Enter website URLs (comma-separated)", placeholder="https://www.uchicago.edu/, https://www.stanford.edu/")
crawl_clicked = st.sidebar.button("Crawl Websites")
file_path = "faiss_store_web.pk1"

# Main placeholder
main_placeholder = st.empty()

# Initialize ChatGroq
llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-8b-8192")

# Function to crawl and scrape website content
def crawl_websites(urls):
    all_text = ""
    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                text = soup.get_text(separator=" ", strip=True)
                all_text += text + "\n"
                st.sidebar.success(f"Successfully crawled: {url}")
            else:
                st.sidebar.error(f"Failed to crawl: {url} (Status code: {response.status_code})")
        except Exception as e:
            st.sidebar.error(f"Error crawling {url}: {str(e)}")
    return all_text

# Process websites
if crawl_clicked:
    if website_urls.strip():
        urls = [url.strip() for url in website_urls.split(",") if url.strip()]
        st.sidebar.success("Website Crawling Started...")
        scraped_content = crawl_websites(urls)

        if scraped_content:
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=55)
            text_chunks = text_splitter.split_text(scraped_content)

            # Create embeddings and FAISS vector store
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_texts(text_chunks, embeddings)

            # Save the FAISS index
            with open(file_path, "wb") as f:
                pickle.dump(vectorstore, f)
            main_placeholder.text("Embedding Vector Built Successfully!")
            st.sidebar.success("Processing Complete!")
        else:
            st.sidebar.error("No content retrieved from the provided URLs.")
    else:
        st.sidebar.error("Please enter at least one website URL.")

# Query Input
query = main_placeholder.text_input("Ask a Question:")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
        retriever = vectorstore.as_retriever()
        chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        # Get response
        result = chain.run(query)
        st.write("### Answer:")
        st.write(result)
    else:
        st.error("Please process the websites first before asking questions.")
