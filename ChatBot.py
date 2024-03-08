import textwrap
import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

class DocumentProcessor:
    def __init__(self, file_path):
        self.loader = TextLoader(file_path)
        self.document = self.loader.load()
    
    def preprocess_text(self, text, width=110):
        lines = text.split('\n')
        wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
        return '\n'.join(wrapped_lines)

    def chunk_documents(self, chunk_size=1000, chunk_overlap=0):
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_documents(self.document)

class TextEmbedder:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings()

    def embed_documents(self, documents):
        db = FAISS.from_documents(documents, self.embeddings)
        return db

class QuestionAnsweringSystem:
    def __init__(self, model_repo_id, model_kwargs):
        self.llm = HuggingFaceHub(repo_id=model_repo_id, model_kwargs=model_kwargs)
        self.qa_chain = None

    def load_qa_chain(self, chain_type):
        self.qa_chain = load_qa_chain(self.llm, chain_type=chain_type)

    def answer_question(self, query, input_documents):
        return self.qa_chain.run(input_documents=input_documents, question=query)

# Usage
file_path = "data.txt"
query = input('.....')

# Document Processing
doc_processor = DocumentProcessor(file_path)
preprocessed_text = doc_processor.preprocess_text(str(doc_processor.document[0]))
chunked_documents = doc_processor.chunk_documents()

# Text Embedding
embedder = TextEmbedder()
embedding_db = embedder.embed_documents(chunked_documents)

# Question Answering
qa_system = QuestionAnsweringSystem(model_repo_id="google/flan-t5-xxl", model_kwargs={'temperature': 0.8, "max_length": 512})
qa_system.load_qa_chain(chain_type="stuff")
answer = qa_system.answer_question(query, embedding_db.similarity_search(query))

print(answer)
