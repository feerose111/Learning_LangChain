from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence , RunnableParallel, RunnablePassthrough , RunnableLambda , RunnableBranch

load_dotenv()

prompt1 = PromptTemplate(
    template= "Write a detailed report on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template= "Write a summary of following text {text}",
    input_variables=['text']
)

model = ChatGroq(model = 'llama-3.1-8b-instant')

parser = StrOutputParser()

report_gen_chain = RunnableSequence(prompt1, model , parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 100, RunnableSequence(prompt2 , model , parser)), # type: ignore
    RunnablePassthrough()
)

final_chain = report_gen_chain | branch_chain

result = final_chain.invoke({'topic': 'Geo-politic standing of Nepal'})

print(result)