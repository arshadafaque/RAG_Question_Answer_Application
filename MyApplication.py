from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

st.title("📚 RAG Application for Question Answering")


@st.cache_resource
def load_vector_store():
    loader = PyPDFLoader("Natural Language Processing-1.pdf")
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(documents)
    
    embedding_model = OllamaEmbeddings(model="mxbai-embed-large:latest")
    vector_store = FAISS.from_documents(split_docs, embedding_model)
    
    return vector_store

vector_store = load_vector_store()
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

try:
    model = OllamaLLM(model="gemma2:2b")
except Exception as e:
    st.error("LLM not found. Please make sure 'gemma2:2b' is pulled using Ollama.")
    st.stop()

# Prompt Template
prompt = PromptTemplate(
    template="""
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, say that you don't know.
Use three sentences maximum and keep the answer concise.

Context:
{context}

Question: {question}
""",
    input_variables=["context", "question"]
)
query = st.chat_input("Ask your question...")


if query:
    parallel_chain = RunnableParallel({
    'context': retriever,
    'question': RunnablePassthrough()
})
    chain = parallel_chain | prompt | model | StrOutputParser()
    with st.spinner("Thinking..."): 
        answer = chain.invoke(query)
        if answer:
            st.write("User ->",query)
            st.write("AI ->",answer)

        
        