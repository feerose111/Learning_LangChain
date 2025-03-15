from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()


model = ChatGoogleGenerativeAI(model = "gemini-1.5-pro")

parser = JsonOutputParser()

prompt_template = PromptTemplate(
    template = 'Give name age and city of a fictional person. \n{format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction' : parser.get_format_instructions()}
)
chain = prompt_template | model | parser

result = chain.invoke({})

print(result)
print(result['name'])