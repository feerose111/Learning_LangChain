from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = GoogleGenerativeAI(model = "gemini-1.5-pro")

prompt1  = PromptTemplate(
    template= "Generate a detail report on {topic}",
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template='Generate 5 line summary on {text}\n',
    input_variables=['text']
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic': 'AI'})
print(result)