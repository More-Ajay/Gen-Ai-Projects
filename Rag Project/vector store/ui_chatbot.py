
from pathlib import Path

import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# -------------------- Configuration --------------------

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "chroma_db"

st.set_page_config(
    page_title="Deep Learning RAG Assistant",
    page_icon="🧠",
    layout="centered"
)

# -------------------- Models and Vector Store --------------------

@st.cache_resource
def load_rag_components():
    embedding_model = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    vectorstore = Chroma(
        persist_directory=str(DB_PATH),
        embedding_function=embedding_model
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    llm = ChatOllama(
        model="llama3.2",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}"""
        )
    ])

    return retriever, llm, prompt


# -------------------- UI --------------------

st.title("🧠 Deep Learning RAG Assistant")
st.caption("Ask questions about your Deep Learning PDF.")

with st.sidebar:
    st.header("About")
    st.write("**Document:** deeplearning.pdf")
    st.write("**LLM:** llama3.2")
    st.write("**Embeddings:** nomic-embed-text")
    st.write("**Search:** MMR")
    st.divider()

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
query = st.chat_input("Ask something about Deep Learning...")

if query:
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching document and generating answer..."):
                retriever, llm, prompt = load_rag_components()

                docs = retriever.invoke(query)

                context = "\n\n".join(
                    doc.page_content for doc in docs
                )

                if not context.strip():
                    answer = "I could not find the answer in the document."
                else:
                    final_prompt = prompt.invoke({
                        "context": context,
                        "question": query
                    })

                    response = llm.invoke(final_prompt)
                    answer = response.content

            st.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        except Exception as e:
            st.error(
                "Unable to generate a response. "
                "Check that Ollama is running, both models are "
                "installed, and the Chroma database exists."
            )
            st.code(str(e))
