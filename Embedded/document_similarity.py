from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv

load_dotenv()

model = HuggingFaceEmbeddings(model_name= "sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = 'tell me about virat kohli.'

doc_embedding = model.embed_documents(documents)
query_embedding = model.embed_query(query)

# Convert embeddings to NumPy arrays
doc_embedding = np.array(doc_embedding)
query_embedding = np.array([query_embedding])

scores = cosine_similarity(query_embedding, doc_embedding).reshape(-1,1)

index , score = sorted(list(enumerate(scores)), key= lambda x:x[1][0])[-1]

print(query)
print(documents[index])
print("similarity score is ", score)