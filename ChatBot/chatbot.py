from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage , SystemMessage
from dotenv import load_dotenv

load_dotenv()

model = GoogleGenerativeAI(model = "gemini-1.5-pro")
chat_history = [
    SystemMessage(content = 'You are a helpful AI assistant')
]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content = user_input)) # type: ignore
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content= result)) # type: ignore
    print('AI: ', result)
    
print(chat_history)