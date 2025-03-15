#This code is run without string output parser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


model = ChatGoogleGenerativeAI(model = "gemini-1.5-pro")
template1 = PromptTemplate(
    template= 'write a detailed report on {topic}',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template= 'write a 5 line summary on the given text. /n {text}',
    input_variables=['text']
)

prompt1 = template1.invoke({'topic': 'flies'})

result = model.invoke(prompt1)

prompt2 = template2.invoke({'text': result.content})

result = model.invoke(prompt2)
print(result.content)