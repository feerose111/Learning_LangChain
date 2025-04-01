from langchain_community.document_loaders import TextLoader

loader = TextLoader('RDBMS.txt')

doc = loader.load()

#print(doc)

print(doc[0].metadata)
