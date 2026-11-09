from  __future__ import annotations
import streamlit as st

from Src.file_utils import read_uploaded_file

from Src.rag_engine import(
    create_documents,
    split_docs,
    build_vectorstore,
    run_jd_matcher,
    generate_complete_report
)


st.set_page_config(
    page_title ="AI Resume JD Matcher",
    page_icon="🎯",
    layout="wide",
)


st.markdown(
        """
    <style>
    .main-title {font-size: 42px; font-weight: 800; margin-bottom: 0px;}
    .subtitle {font-size: 18px; color: #666; margin-bottom: 25px;}
    .stage-box {padding: 14px; border-radius: 12px; background: #f7f7f7; border: 1px solid #e8e8e8; margin-bottom: 10px;}
    .success-box {padding: 12px; border-radius: 10px; background: #eaffea; border: 1px solid #8de28d;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🎯 RAG Resume JD Matcher</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Resume + Job Description → RAG Pipeline → Skill Gap Analysis, Resume Suggestions & Interview Prep</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ RAG Settings")
    chunk_size = st.slider("Chunk size", 300, 1500, 800, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 150, 50)
    st.divider()
    st.markdown("### RAG Stages")
    st.markdown("1. Load Resume & JD")
    st.markdown("2. Split into Chunks")
    st.markdown("3. Convert to Embeddings")
    st.markdown("4. Store in ChromaDB")
    st.markdown("5. Retrieve Relevant Context")
    st.markdown("6. Generate Career Advice")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Upload Resume")
    resume_file = st.file_uploader("Upload resume (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"], key="resume")
    resume_text_input = st.text_area("Or paste resume text", height=220, key="resume_text")

with col2:
    st.subheader("💼 Upload Job Description")
    jd_file = st.file_uploader("Upload JD (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"], key="jd")
    jd_text_input = st.text_area("Or paste job description", height=220, key="jd_text")

resume_text = read_uploaded_file(resume_file) if resume_file else resume_text_input
jd_text = read_uploaded_file(jd_file) if jd_file else jd_text_input

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chunks" not in st.session_state:
    st.session_state.chunks =[]

st.divider()

if st.button("🚀 RAG Resume JD Checker Index", type="primary"):
    if not resume_text.strip() or not jd_text.strip():
        st.error("Please upload or paste both Resume and Job Description.")
    else:
        with st.spinner("Running RAG stages: loading → chunking → embeddings → vector database..."):
            docs = create_documents(resume_text, jd_text)
            chunks = split_docs(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap )
            vectorstore = build_vectorstore(chunks)

            st.session_state.vectorstore = vectorstore
            st.session_state.chunks = chunks

        st.success("RAG index created successfully!")
        c1, c2, c3 = st.columns(3)
        c1.metric("Documents", "2")
        c2.metric("Chunks", len(st.session_state.chunks))
        c3.metric("Vector DB", "ChromaDB")
if st.session_state.vectorstore:
    st.markdown("### ✅ Ask Career Questions")

    quick_questions = [
        "How well does this resume match the job description?",
        "What are the missing skills for this role?",
        "How can I improve this resume?",
        "Suggest 3 projects to become suitable for this role.",
        "Generate interview questions based on the skill gaps.",
    ]
    selected = st.selectbox("Choose a question", quick_questions)
    custom_question = st.text_input("Or ask your own question")
    final_question = custom_question if custom_question else selected

    if st.button("🤖 Get Career Coach Answer"):
            with st.spinner("Retrieving context and generating answer..."):
                answer, sources = run_jd_matcher(
                    st.session_state.vectorstore,
                    resume_text,
                    jd_text,
                    final_question,
                )
    
            st.markdown("## 🎯 Career Coach Response")
            st.write(answer)
    
            with st.expander("🔍 Retrieved Context Used by RAG"):
                for i, doc in enumerate(sources, 1):
                    st.markdown(f"#### Source Chunk {i}")
                    st.caption(str(doc.metadata))
                    st.write(doc.page_content[:1200])
    
    st.divider()
    
    if st.button("📊 Generate Complete Career Report"):
        with st.spinner("Generating complete RAG-based career report..."):
            report, sources = generate_complete_report(
                    st.session_state.vectorstore,
                    resume_text,
                    jd_text,
                )
        st.markdown("## 📊 Complete Career Report")
        st.write(report)
    
else:
    st.info("Upload/paste Resume and Job Description, then click 'Build Career Coach RAG Index'.")
    
st.divider()
st.markdown("### 🧠 How this Traditional RAG Project Works")
st.markdown(
        """
    1. **Load documents:** Resume and Job Description are converted into LangChain `Document` objects.  
    2. **Chunking:** Large text is split using `RecursiveCharacterTextSplitter`.  
    3. **Embeddings:** Each chunk is converted into a numerical vector using HuggingFace embeddings.  
    4. **Vector Database:** Chunks and embeddings are stored inside ChromaDB.  
    5. **Query Embedding:** User question is also converted into an embedding.  
    6. **Similarity Search:** ChromaDB finds the most relevant resume/JD chunks.  
    7. **LLM Generation:** Groq Llama model generates the final career guidance using only retrieved context.  
    """
    )
    
         