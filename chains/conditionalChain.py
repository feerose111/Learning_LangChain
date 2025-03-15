from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnableBranch , RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatGroq(model= "llama-3.1-8b-instant")

parser1 = StrOutputParser()

class Feedback(BaseModel): # type: ignore
    
    sentiment : Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback.')
    
parser2 = PydanticOutputParser(pydantic_object= Feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment of following feedbacktext into positive and negative \n {feedback} \n {format_instruction} ",
    input_variables=["feedback"],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

prompt2 = PromptTemplate(
    template = "Generate an appropriate response to this positive feedback \n {feedback}",
    input_variables= ['feedback']
)
prompt3 = PromptTemplate(
    template = "Generate an appropriate response to this negative feedback \n {feedback}",
    input_variables= ['feedback']
)

classifier_chain = prompt1 | model | parser2
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant")

parser1 = StrOutputParser()

class Feedback(BaseModel):
    
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback.')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment of following feedback text into positive and negative \n {feedback} \n {format_instruction} ",
    input_variables=["feedback"],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)
classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template="Give any appropriate response to this positive feedback \n {feedback}",
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template="Give any appropriate response to this negative feedback \n {feedback}",
    input_variables=['feedback']
)

branch_chain = RunnableBranch (
    (lambda x: x.sentiment == 'positive', prompt2 | model | parser1), # type: ignore
    (lambda x: x.sentiment == 'negative', prompt3 | model | parser1), # type: ignore
    RunnableLambda(lambda x : "couldnt find any sentiment")
)

chain = classifier_chain | branch_chain

output = chain.invoke({"feedback":"I loved the service"})

print(output)
