# 🚗 EcoTrans Assistant: Intelligent Question Answering System
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)

EcoTrans Assistant is an intelligent system that helps users find answers to questions related to car rentals using a knowledge base and natural language processing (NLP) technologies. The system uses text embeddings, graph models, and the DeepSeek API to analyze queries and provide accurate responses.

---
## 📋 Table of Contents
1. [Project Description](#project-description)
2. [Technologies](#technologies)
---
## 🌟 Project Description

EcoTrans Assistant is designed to automate the process of answering frequently asked questions from users. The system:
- Uses a pre-prepared knowledge base (`knowledge_base.py`) to search for similar questions.
- Generates text embeddings using the `SentenceTransformer` model.
- Analyzes user queries through a graph model and the DeepSeek API.
- Returns the most relevant answer or notifies if no answer is found.

The project is ideal for companies providing car rental services, such as EcoTrans, to improve customer interaction.

---
## 💻 Technologies

The project uses the following technologies and libraries:
- **Python 3.8+**
- **LangChain**: For working with graph models and tools.
- **SentenceTransformers**: For creating text embeddings.
- **FAISS**: For efficient similarity search of questions.
- **DeepSeek API**: For generating responses based on text analysis.
---