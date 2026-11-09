import os
import shutil
from  pathlib import Path
from typing import List,Tuple
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DB_dir = "Chroma DB"

def get_llm(model : str ="qwen/qwen3.8-27b", temperature: float =0.2):
    try:
        api_key = os.getenv("GROQ_API_KEY")
    except:
        print("Model not loaded")

    return ChatGroq(model=model,temperature=temperature)


def get_embeddings():
    return HuggingFaceEmbeddings(model = "sentence-transformers/all-MiniLM-L6-v2")


def load_text_file(file_path:str, source_name:str, doc_type:str):
    path = Path(file_path)
    text = path.read_text(encoding='utf-8', errors='ignore')
    return [Document(page_content=text, 
                     metadata ={"source":source_name,"doc_type":doc_type})]


def create_documents (resume_text:str, jd_text:str) -> list[Document]:
  
    return [Document(page_content=resume_text, 
                     metadata ={"source":"uploaded_resume","doc_type":"resume"}),
            
            Document(page_content=jd_text, 
                     metadata ={"source":"uploaded_jd","doc_type":"JD"}) ]


def split_docs(docs:List[Document], chunk_size:int =1000,
                    chunk_overlap:int = 150)->List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size,
                                              chunk_overlap=chunk_overlap,
                                              separators=['\n\n',"\n","."," "])
    return splitter.split_documents(docs)


# def build_vectorstore(chunks: List[Document], persist_directory:str =DB_dir):
#     if Path(persist_directory).exists():
#         shutil.rmtree(persist_directory)

#     Vectorstore = Chroma.from_documents(documents = chunks,
#                                         embedding= get_embeddings(),
#                                         persist_directory=persist_directory,
#                                         collection_name="rag_jd_matcher"
#                                         )
#     return Vectorstore
import uuid
from pathlib import Path

def build_vectorstore(chunks: List[Document], persist_directory: str = DB_dir):
    # Create a unique sub-folder inside your directory every time you run
    # This completely bypasses the Windows / OneDrive "Access is Denied" file lock!
    unique_id = str(uuid.uuid4())[:8]
    fresh_directory = str(Path(persist_directory) / f"session_{unique_id}")

    Vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=fresh_directory,
        collection_name="rag_jd_matcher"
    )
    
    return Vectorstore


def retrieve_context(vectorstore,query:str, k:int =3)-> Tuple[str, List[Document]]:
    retriever = vectorstore.as_retriever(search_Kwargs={'k':k} )
    docs = retriever.invoke(query)
    context = "\n\n".join([f"SOURCE: {d.metadata}\nCONTENT:\n{d.page_content}" for d in docs])
    return context,docs 


def run_jd_matcher(vectorstore,resume_text:str, jd_text:str, question:str):
    llm =get_llm()
    retrieval_query = f'''Resume content and JD content relavant to the career coaching question:
    {question}'''
    context, source_docs = retrieve_context(vectorstore,retrieval_query, k=3)
   
    prompt = ChatPromptTemplate.from_template("""
    You are an expert AI Career Coach for students, freshers and working professionals.
    Use ONLY the given context from the resume and job description.
    Do not invent skills, experience or job requirements.
    
    CONTEXT:
    {context}
    
    USER QUESTION:
    {question}
    
    Give a clear, practical answer with these sections when relevant:
    1. Current Match Summary
    2. Strengths
    3. Missing Skills / Gaps
    4. Recommended Improvements
    5. Suggested Projects
    6. Interview Preparation Tips
    
    Keep the answer simple, actionable and beginner-friendly.
    """)

    chain = prompt| llm | StrOutputParser()

    answer = chain.invoke({"context":context,"question":question})

    return answer,source_docs


def generate_complete_report(vectorstore, resume_text: str, jd_text: str):
    question = """
    Analyze this resume against this job description. Provide ATS-style score, skill match, missing skills,
    resume improvement suggestions, project suggestions, and interview questions.
    """
    return run_jd_matcher(vectorstore, resume_text, jd_text, question)


    