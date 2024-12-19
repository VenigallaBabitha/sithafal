import os
import pickle
import time
from pdfminer.high_level import extract_text
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

#Streamlit UI

st.title("Task-1: Chat with PDF Using RAG Pipeline")
st.sidebar.title("PDF Cracker")

#File uploader

uploaded_files = st.sidebar.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])
process_pdf_clicked=st.sidebar.button("Process PDFs")
file_path = "faiss_store_openai.pk1"

#Main placeholder
main_placeholder=st.empty()

#Initialize ChatGroq

groq_api_key ="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" # Replace with your valid Groq API key

llm =ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-8b-8192")

#Process PDFS
if process_pdf_clicked:
    if uploaded_files:
        st.sidebar.success("Text Extraction Started...")
        all_text=" "
        for uploaded_file in uploaded_files:
            extracted_text = extract_text(uploaded_file)
            all_text +=extracted_text + "\n"

            #Split text into chunks
            text_splitter =RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=55)
            text_chunks= text_splitter.split_text(all_text)

            #Create enbeddings and FAISS vector store
            embeddings =HuggingFaceEmbeddings (model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore= FAISS.from_texts(text_chunks, embeddings)

            #Save the FAISS index
            with open(file_path, "wb") as f:
                pickle.dump(vectorstore, f)
            main_placeholder.text("Embedding Vector Built Successfully!")
            st.sidebar.success("Processing Complete!")
        else:
            st.sidebar.error("Please upload at least one PDF file.")

#Query Input

query=main_placeholder.text_input("Ask a Question:")
if query:
    if os.path.exists(file_path):
        with open(file_path,"rb")as f:
            vectorstore= pickle.load(f)
        retriever= vectorstore.as_retriever()
        chain= RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        #Get response
        result =chain.run(query)
        st.write("### Answer:")
        st.write(result)
    else:
        st.error("Please process the PDFs first before asking questions.")