# Paper RAG ChatBot with Llama2 and Gradio
PDFChatBot is a Python-based chatbot designed to answer questions based on the content of uploaded PDF files. It utilizes the Gradio library for creating a user-friendly interface and LangChain for natural language processing.
Technologies Used:
* Langchain
* Llama2
* ChromaDB
* Hugging Face
* Gradio

Before running the ChatBot, ensure that you have the required dependencies installed. You can install them using the following command:
```
pip install -r requirements.txt
```
The ChatBot uses a configuration file (config.yaml) to specify Hugging Face model and embeddings details. Make sure to update the configuration file with the appropriate values if you wanted to try another model or embeddings. The default LLM used if Llama2-13b
1. Upload a PDF file using the "📁 Upload PDF" button.
2. Enter your questions in the text box.
3. Click the "Send" button to submit your question.
4. View the chat history and responses in the interface.

To run locally, run the following command:
```
cd src
python app.py
```
