# 🚀 LangChain & LLM Experiments

## 📌 Overview

This repository contains my experiments and learning progress with **LangChain**, focusing on integrating **Large Language Models (LLMs)** for chatbot development, prompt engineering, embedding models, and output parsing.

## 🛠️ Technologies Used

- **LangChain** 🦜🔗
- **Python** 🐍
- **LLM Providers**: OpenAI, Google Gemini, Hugging Face, Anthropic
- **Git & GitHub**
- **Virtual Environments (venv)**

## 📂 Project Structure

```
LangChain/
│── chains/               # Different types of LangChain chains (sequential, parallel, conditional)
│── ChatBot/              # Chatbot implementations using LLMs
│── ChatModels/           # Various chat models (Google, OpenAI, Hugging Face, Anthropic)
│── Embedded/             # Document similarity & embeddings
│── LangChainPrompts/     # Prompt engineering experiments
│── LLMs/                 # General LLM-related scripts
│── OutputParsers/        # JSON, Pydantic, and structured output parsers
│── langchainenv/         # Virtual environment (excluded from Git)
│── .env                  # API keys & environment variables (excluded from Git)
│── requirements.txt      # Dependencies
│── README.md             # Project documentation
```

## 🚀 Setup & Installation

### **1️⃣ Clone the Repository**

```sh
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### **2️⃣ Create & Activate Virtual Environment**

```sh
# Windows
python -m venv langchainenv
langchainenv\Scripts\activate

# Mac/Linux
python3 -m venv langchainenv
source langchainenv/bin/activate
```

### **3️⃣ Install Dependencies**

```sh
pip install -r requirements.txt
```

### **4️⃣ Set Up API Keys**

Create a `.env` file and add your API keys:

```
OPENAI_API_KEY=your-api-key
GOOGLE_API_KEY=your-api-key
```

### **5️⃣ Run a Script**

```sh
python ChatModels/chatmodel_google.py
```

## ✅ Features Implemented

✔️ Integrated OpenAI, Google, Hugging Face, and Anthropic LLMs\
✔️ Created **chatbots** using LangChain\
✔️ Implemented **chaining mechanisms** (sequential, parallel, conditional)\
✔️ Developed **custom prompt templates**\
✔️ Implemented **document embeddings & similarity**\
✔️ Used **structured output parsers** (JSON, Pydantic)\



