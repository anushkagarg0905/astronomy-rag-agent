import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA




DATA_PATH = "data"

def load_documents():
    documents = []
    for file in os.listdir(DATA_PATH):
        if file.lower().endswith(".pdf"):
            path = os.path.join(DATA_PATH, file)
            print("Loading:", path)
            loader = PyPDFLoader(path)
            documents.extend(loader.load())
    return documents




def create_vector_db():
    docs = load_documents()
    print("Loaded documents:", len(docs))

    if len(docs) == 0:
        raise ValueError("No documents found in data folder")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    splits = text_splitter.split_documents(docs)
    print("Chunks created:", len(splits))

    if len(splits) == 0:
        raise ValueError("Text splitter produced zero chunks")

    embeddings = OllamaEmbeddings(
        model="mxbai-embed-large",
        base_url="http://127.0.0.1:11434"
    )

    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    vectordb.persist()
    return vectordb

def main():
    vectordb = create_vector_db()

    llm = OllamaLLM(model="tinyllama")

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever()
    )

    while True:
        query = input("Ask: ")
        if query == "exit":
            break
        result = qa.invoke(query)
        print(result["result"])



if __name__ == "__main__":
    main()