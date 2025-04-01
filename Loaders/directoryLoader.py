from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
parser = StrOutputParser()
loader  = DirectoryLoader(
    path = 'books',
    glob= '*.pdf',
    loader_cls = PyPDFLoader # type: ignore
)
doc = loader.load()

prompt= PromptTemplate(
    template= "Write a insightful and detailed summary on the  {topic} , from the following text\n{text}",
    input_variables=['topic', 'text']
)


model = ChatGroq(model = 'llama-3.1-8b-instant')

text = []
for i in range(len(doc)-320):
    text.append(doc[i].page_content)
    

#doc = loader.lazy_load() for faster loading and folder with large amount of files
# print(len(doc))
# print(doc[300].page_content)
# print(doc[300].metadata)

chain = prompt | model | parser

result = chain.invoke({'topic': 'list 5 human haitual traits', 'text': text})

print(result)