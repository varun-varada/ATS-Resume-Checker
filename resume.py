# from langchain.google_genai import ChatGoogleGenerativeAI 
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import tempfile
google_api=os.getenv('GOOGLE_API_KEY')
llm=ChatGoogleGenerativeAI(google_api_key=google_api,model="gemini-2.5-flash")

st.title("Resume ↔ Job Description Match ")
job_title=st.text_input("Enter the Job Title")
knowledge_base=st.file_uploader("Upload Resume ",type=['pdf','docx','txt'])
suggestions=st.selectbox("Add Suggestions and improvements ?",options=["Yes","No"])
description=st.text_area("Enter the Job Description")
button=st.button("Get Match Score")
if button:
    if not job_title.strip() and knowledge_base is None and not description.strip():
        st.warning("Please provide job Title, resume file and job description.")
    with tempfile.NamedTemporaryFile(delete=False,suffix=os.path.splitext(knowledge_base.name)[1]) as tmp_file:
        tmp_file.write(knowledge_base.read())
        tempfile_path=tmp_file.name 
    if knowledge_base.name.endswith('.pdf'):
        loader=PyMuPDFLoader(tempfile_path)
        docs=loader.load()
    else:
        st.error("Unsupported file type. Please upload a PDF file.")
        st.stop()
    
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=200)
    chunks=text_splitter.split_documents(docs)
    
    embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector=FAISS.from_documents(chunks,embedding)
    
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    
    if suggestions=="Yes":
        suggestions="Yes, short provide suggestions and improvements to enhance the resume's alignment with the job description."
        
    prompt_template=f"""
   You are an expert HR professional And ATS Checker . Your task is to evaluate how well a candidate's resume matches a specific job description.
Generate a match score between 1 and 100 \n\n , state a sentiment (Positive/Neutral/Negative), and provide a brief explanation.

Consider the following criteria:
1. Relevance of Skills
2. Experience
3. Education
4. Achievements / Certifications
5. Overall Fit

Use the following format exactly:
Match Score: <score between 1-100 \n>
Sentiment: <Positive/Neutral/Negative>
Explanation: <brief explanation>

Job Title: {job_title}
Job Description: {description}
\n\n\n
Suggestions and Improvements: {suggestions}
Resume Content: {{resume_content}}

    """
    prompt=PromptTemplate(
        input_variables=["resume_content"],
        template=prompt_template
    )
    retriever=vector.as_retriever(search_type="similarity",search_kwargs={"k":3})
    qa_chain=RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=retriever,chain_type_kwargs={"prompt":prompt,"document_variable_name": "resume_content" })
    
    with st.spinner("Calculating Match Score..."):
         result=qa_chain.run(job_title)
    st.subheader("Match Score Result")
    st.write(result)
    
    