from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

documents = [
    'Kathmandu is the capital of Nepal.',
    'Kathmandu is a district in Nepal.',
    'Kathmandu is a city of Nepal.',
]

result = embedding.embed_documents(documents)

print(str(result))