import streamlit as st
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq

st.set_page_config(page_title="Chatbot RAG Excel", layout="wide")

# Fungsi bina database vektor (RAG)
@st.cache_resource
def prepare_data():
    df = pd.read_excel("data.xlsx")
    df['combined'] = df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    loader = DataFrameLoader(df, page_content_column="combined")
    docs = loader.load()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

st.title("🤖 Chatbot Data Excel (Groq RAG)")

# Inisialisasi
if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    vectorstore = prepare_data()
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Tanya soalan mengenai data anda:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # Cari data relevan
        docs = vectorstore.similarity_search(prompt, k=3)
        context = "\n".join([d.page_content for d in docs])
        
        # Panggil Groq
        chat_completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": f"Jawab berdasarkan data ini sahaja: {context}"},
                {"role": "user", "content": prompt}
            ]
        )
        
        answer = chat_completion.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)

except Exception as e:
    st.info("Sila pastikan data.xlsx sudah dimuat naik dan API Key telah dimasukkan di Settings.")
