from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("document loaders/notes.pdf")

docs = data.load()

print(docs[0])