from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from knowledge_base import prepared_answers
import json


class CustomEmbeddings(Embeddings):
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_tensor=False)[0].tolist()


def search_similar_questions(query, k=1):
    results = vectorstore.similarity_search(query, k=k)
    new_results = [result.page_content for result in results]

    return new_results[k-1]


def update_vectorstore(new_questions_list, new_embeddings):
    new_vectorstore = FAISS.from_texts(texts=new_questions_list, embedding=new_embeddings)

    return new_vectorstore


embeddings = CustomEmbeddings()
vectorstore = FAISS.from_texts(texts=[key+' '+value for key, value in prepared_answers.items()], embedding=embeddings)
