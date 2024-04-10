from llama_index.llms.mistralai import MistralAI
from llama_index.llms.azure_openai import AzureOpenAI

from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.embeddings.openai import (
    OpenAIEmbedding,
    OpenAIEmbeddingMode,
    OpenAIEmbeddingModelType,
)

from llama_index.core.settings import Settings
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
import gradio as gr
from gradio_pdf import PDF
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv('AZURE_OPENAI_API_KEY')
azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
llm = AzureOpenAI(api_key=api_key, engine="gpt-35-turbo", model="gpt-35-turbo", azure_endpoint=azure_endpoint)
embed_model = AzureOpenAIEmbedding(azure_deployment='text-embedding-ada-002', model=OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002, api_key=api_key, azure_endpoint=azure_endpoint)

Settings.llm = llm
Settings.embed_model = embed_model

def qa(question: str, doc: str) -> str:
    my_pdf = SimpleDirectoryReader(input_files=[doc]).load_data()
    my_pdf_index = VectorStoreIndex.from_documents(my_pdf)
    my_pdf_engine = my_pdf_index.as_query_engine()
    response = my_pdf_engine.query(question)
    return response

demo = gr.Interface(
    qa,
    [gr.Textbox(label="Question"), PDF(label="Document")],
    gr.Textbox())

if __name__ == "__main__":
    demo.launch()
