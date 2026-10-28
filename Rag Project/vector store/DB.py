from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document

docs = [
        Document(page_content="Python is widely used in Artificial Intelligence." , metadata={"source": "AI_Book"}),
        Document(page_content="Pandas is used for data analysis in Python." , metadata={"source": "Datascience_Book"}),
        Document(page_content="Neural network are used in Deep Learning." , metadata={"source": "DL_Book"}),
]

embedding_model = OllamaEmbeddings(model="nomic-embed-text")


vector_store = Chroma.from_documents(
    documents = docs,
    embedding = embedding_model,
    persist_directory = "chroma_db"
)

result = vector_store.similarity_search("What is used for data analysis?" , k=2)

for r in result:
    print(r.page_content)
    print(r.metadata)

retriever = vector_store.as_retriever()

docs = retriever.invoke("Explain Deep Learning?")

for d in docs:
    print(d.page_content)