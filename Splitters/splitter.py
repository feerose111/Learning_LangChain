from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, Language

loader = PyPDFLoader('syllabus.pdf')

doc = loader.load()

# splitter = CharacterTextSplitter(
#     chunk_size = 200,
#     chunk_overlap = 40,
#     separator = ''
# )

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 0,
)

# splitter = RecursiveCharacterTextSplitter.from_language(  #for splitting a python code
#     language = Language.PYTHON,
#     chunk_size = 350,
#     chunk_overlap = 0,
# )

result = splitter.split_documents(doc)

for document in result:
    print(document.page_content)