from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence , RunnableParallel , RunnablePassthrough , RunnableLambda

load_dotenv()

model = ChatGroq(model = 'llama-3.1-8b-instant')

prompt1 = PromptTemplate(
    template= "Generate a joke on {topic}\n",
    input_variables=["topic"]
)

def word_count(text):
    return len(text.split())

parser  = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1 , model , parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word count':RunnableLambda(word_count)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)
result = final_chain.invoke({'topic':'AI'})

final_result = """Joke :{}\nTotal word count: {}""".format(result['joke'], result['word count'])
print(final_result)
