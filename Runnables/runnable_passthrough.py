from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence , RunnableParallel , RunnablePassthrough

load_dotenv()

model = ChatGroq(model = 'llama-3.1-8b-instant')

prompt1 = PromptTemplate(
    template= "Generate a joke on {topic}\n",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="\n Write a very short explanation of following {joke}",
    input_variables=['joke']
)
parser  = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1 , model , parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation':RunnableSequence(prompt1, model, parser , prompt2, model , parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)
result = final_chain.invoke({'topic':'AI'})
print(result['joke'], result['explanation'])
