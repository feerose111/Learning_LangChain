from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


model = ChatGoogleGenerativeAI(model = "gemini-1.5-pro")

class person(BaseModel):
    
    name : str = Field(description='Name of a person')
    age : int = Field(gt=18, description='Age of the person')
    city : str  = Field(description='Name of the city the person lives in')

parser = PydanticOutputParser(pydantic_object=person)

template = PromptTemplate(
    template = 'Give name , age and city of fictional person of {place} \n {format_instruction}',
    input_variables=['place'],
    partial_variables= {'format_instruction' : parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'place':'tajikistan'})
print(result)