from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# 1. Prompt Template
prompt = PromptTemplate.from_template("Explain {topic} in simple words")


# 2. Model
model = ChatOllama(model="llama3.2")


# 3. Output Parser
parser = StrOutputParser()


chain = prompt | model | parser


result = chain.invoke("Machine Learning")


print(result)