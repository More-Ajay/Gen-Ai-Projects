from langchain_ollama import ChatOllama

from dotenv import load_dotenv

load_dotenv()

model = ChatOllama(
    model="llama3.2",
    temperature=0
)

print("-------------Welcome! type 0 to exit the application--------------")

while True:
    prompt = input("You: ")
    
    if prompt == "0":
        break

    response = model.invoke(prompt)

    print("Bot: ",response.content)