import yaml
import fitz
import torch
import gradio as gr
from PIL import Image
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class PDFChatBot:
    def __init__(self, config_path="../config.yaml"):
        """
        Initialize chatbot 

        Parameters:
            config_path(str): path to model config files
        """
        self.processed = False
        self.page = 0
        self.chat_history = []
        self.config = self.load_config(config_path)
        # Initialize other attributes to None
        self.prompt = None
        self.documents = None
        self.embeddings = None
        self.vectordb = None
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.chain = None

    def load_config(self, file_path):
        """
        Load configuration from a YAML file.

        Parameters:
            file_path (str): Path to the YAML configuration file.

        Returns:
            dict: Configuration as a dictionary.
        """
        with open(file_path, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
                return config
            except yaml.YAMLError as exc:
                print(f"Error loading configuration: {exc}")
                return None
            
    def add_text(self, history, text):
        """
        Add user-entered text to the chat history.

        Parameters:
            history (list): List of chat history tuples.
            text (str): User-entered text.

        Returns:
            list: Updated chat history.
        """
        if not text:
            raise gr.Error('Enter text')
        history.append((text, ''))
        return history
    

    def create_prompt_template(self):
        """
        Create a prompt template for the chatbot.
        """
        # template = (
        #     f"The assistant should provide detailed explanations."
        #     "Combine the chat history and the question into your answer."
        # )
        template = """Given the following conversation and a follow-up message, \
        rephrase the follow-up message to a stand-alone question or instruction that \
        represents the user's intent, add context if necessary to generate a complete and \
        unambiguous question or instruction, only based on the history, don't make up messages. \
        Maintain the same language as the follow up input message.

        Chat History:
        {chat_history}

        Follow Up Input: {question}
        Standalone question or instruction:"""
        # self.prompt = PromptTemplate.from_template(template=template, input_variables=["chat_history", "question"])
        self.prompt = PromptTemplate(
            template=template,
            input_variables=[
                "chat_history",
                "question",
            ],
        )

    def load_embeddings(self):
        """
        Load embeddings from Hugging Face and set in the config file.
        """
        self.embeddings = HuggingFaceEmbeddings(model_name=self.config.get("modelEmbeddings"))


    def load_vectordb(self):
        """
        Load the vector database from the documents and embeddings.
        """
        self.vectordb = Chroma.from_documents(self.documents, self.embeddings)

    def load_tokenizer(self):
        """
        Load the tokenizer from Hugging Face and set in the config file.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.get("autoTokenizer"))

    def load_model(self):
        """
        Load the causal language model from Hugging Face and set in the config file.
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.get("autoModelForCausalLM"),
            device_map='auto',
            torch_dtype=torch.float32,
            token=True,
            load_in_8bit=False
        )

    def create_pipeline(self):
        """
        Create a pipeline for text generation using the loaded model and tokenizer.
        """
        pipe = pipeline(
            model=self.model,
            task='text-generation',
            tokenizer=self.tokenizer,
            max_new_tokens=200
        )
        self.pipeline = HuggingFacePipeline(pipeline=pipe)

    def create_chain(self):
        """
        Create a Conversational Retrieval Chain
        """
        self.chain = ConversationalRetrievalChain.from_llm(
            self.pipeline,
            chain_type="stuff",
            retriever=self.vectordb.as_retriever(search_type="similarity",search_kwargs={'k': 2}),
            condense_question_prompt=self.prompt,
            return_source_documents=True
        )

    def process_file(self, file):
        """
        Process the uploaded PDF file and initialize necessary components: Tokenizer, VectorDB and LLM.

        Parameters:
            file (FileStorage): The uploaded PDF file.
        """
        self.create_prompt_template()
        self.documents = PyPDFLoader(file.name).load_and_split(CharacterTextSplitter(chunk_size=200, chunk_overlap=0))
        self.load_embeddings()
        self.load_vectordb()
        self.load_tokenizer()
        self.load_model()
        self.create_pipeline()
        self.create_chain()

    def generate_response(self, history, query, file):
        """
        Generate a response based on user query and chat history.

        Parameters:
            history (list): List of chat history tuples.
            query (str): User's query.
            file (FileStorage): The uploaded PDF file.

        Returns:
            tuple: Updated chat history and a space.
        """
        if not query:
            raise gr.Error(message='Submit a question')
        if not file:
            raise gr.Error(message='Upload a PDF')
        if not self.processed:
            self.process_file(file)
            self.processed = True

        result = self.chain({"question": query, 'chat_history': self.chat_history}, return_only_outputs=True)
        self.chat_history.append((query, result["answer"]))
        self.page = list(result['source_documents'][0])[1][1]['page']

        for char in result['answer']:
            history[-1][-1] += char
        return history, " "
    
    def render_file(self, file):
        """
        Renders a specific page of a PDF file as an image.

        Parameters:
            file (FileStorage): The PDF file.

        Returns:
            PIL.Image.Image: The rendered page as an image.
        """
        doc = fitz.open(file.name)
        page = doc[self.page]
        pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
        image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        return image