from __future__ import annotations
import tempfile
from pathlib import Path
from typing import Optional
from pypdf import PdfReader
import docx2txt 


def read_uploaded_file(uploaded_file)-> str:
    ''' read txt, pdf, pr docs uploaded from streamlit'''
    if uploaded_file is None:
        return ""
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix =='.txt':
        return uploaded_file.read().decode("utf-8",errors='ignore')

    with tempfile.NamedTemporaryFile(delete=False,suffix=suffix) as temp:
        temp.write(uploaded_file.getvalue())
        temp_path = temp.name

    if suffix == '.pdf':
        reader = PdfReader(temp_path)
        text =[]
        for page in reader.pages:
            text.append(page.extract_text() or "")
        return "\n".join(text)

    if suffix =='.docx':
        return docx2txt.process(temp_path)

    raise ValueError("Unsupported file type. Please upload .txt, .pdf, or .docx")