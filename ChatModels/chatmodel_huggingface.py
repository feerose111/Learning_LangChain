from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# Load the model and endpoint
repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    temperature=0.5,
    task = 'text-generation') # type: ignore

model  = ChatHuggingFace(llm = llm, model = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')

print (model.invoke('what is capital of nepal?').content)