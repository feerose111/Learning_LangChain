from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel

load_dotenv()

prompt1 = PromptTemplate(
    template="Write a tweet on a {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Write a linkedin post on a {topic}",
    input_variables=['topic']
)

model = ChatGroq(model = 'llama-3.1-8b-instant')
 
parser = StrOutputParser()

chain = RunnableParallel({
    'tweet ': RunnableSequence(prompt1, model ,parser ),
    'linkedin post': RunnableSequence(prompt2, model, parser)    
})

result = chain.invoke({'topic': 'AI'})
print(result)

