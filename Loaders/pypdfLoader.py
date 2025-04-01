from langchain_community.document_loaders import PyPDFLoader


loader = PyPDFLoader('syllabus.pdf')

doc = loader.load()

print(doc)
print(doc[0].metadata)
print(doc[0].page_content)