from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence

load_dotenv()

model = ChatGroq(model = 'llama-3.1-8b-instant')

prompt1 = PromptTemplate(
    template= "Generate a joke on {topic}\n",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Write a  short explanation of following {joke}",
    input_variables=['joke']
)
parser  = StrOutputParser()

chain1 = RunnableSequence( prompt1, model, parser )
chain2 = RunnableSequence( prompt1, model, parser , prompt2, model , parser)

result1 = chain1.invoke({'topic': 'AI'})
result2 = chain2.invoke({'topic': 'AI'})

print(result1)
print(result2)