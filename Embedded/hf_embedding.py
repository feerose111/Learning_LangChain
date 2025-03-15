from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

model = HuggingFaceEmbeddings(model_name= "sentence-transformers/all-MiniLM-L6-v2")

result = model.embed_query("hi my name is john")

print(str(result))