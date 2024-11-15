from langchain.chains import RetrievalQA
from langchain_unstructured import UnstructuredLoader
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

file_paths = [
    "cinderella.txt",
]

loader = UnstructuredLoader(file_paths)
docs = loader.load()

text_spliter = CharacterTextSplitter(
    separator="\n",
    chunk_size=2000,
    chunk_overlap=200
)

texts = text_spliter.split_documents(docs)
embeddings = HuggingFaceEmbeddings()

db = FAISS.from_documents(texts, embeddings)
model = OllamaLLM(model="llama3:8b")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful assistant. 
            Answer questions using only the following context. 
            If you don't know the answer just say you don't know, don't make it up:
            \n\n
            {context}",
            """
        ),
        ("human", "{question}"),
    ]
)

retriever = db.as_retriever()

# chain = (
#     {
#         "context": retriever,
#         "question": RunnablePassthrough(),
#     }
#     | prompt
#     | model
# )

query = "Who killed Cinderella?"

qa = RetrievalQA.from_chain_type(llm=model
                        , chain_type = 'stuff'
                        , retriever = db.as_retriever(
                            search_type='mmr'
                           ,search_kwargs = {'k':1,'fetch_k':3})
                        , return_source_documents = True
                        , chain_type_kwargs={"prompt": prompt})
result = qa.invoke(query)

print(result)
