from dotenv import load_dotenv

load_dotenv()

# from langchain.chat_models import init_chat_model

# model = init_chat_model("gemini-3.7-flash" , model_provider="google_genai")

# response = model.invoke("What is cricket?")

# print(response)


from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

response = model.invoke("What is cricket?")

print(response)
