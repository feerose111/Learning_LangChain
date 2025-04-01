from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

url = 'https://www.sharesansar.com/news-page'

loader = WebBaseLoader(url)

doc = loader.load()

#print(doc[0].page_content)

prompt1 = PromptTemplate(
    template= "Answer the following question \n{question} from the following text \n{text}",
    input_variables=['question', 'text']
)

model = ChatGroq(model = 'llama-3.1-8b-instant')

parser = StrOutputParser()

chain = prompt1 | model | parser

result = chain.invoke({'question': 'what is the latest news of today? ', 'text': doc[0].page_content})

print(result)