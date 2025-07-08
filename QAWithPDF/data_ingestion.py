from llama_index.core import SimpleDirectoryReader
import sys
from QAWithPDF.exception import customexception
from QAWithPDF.logger import logging

def load_data(data_dir):
    """
    Load PDF documents from a specified directory.

    Parameters:
    - data_dir (str): The path to the directory containing PDF files.

    Returns:
    - A list of loaded PDF documents.
    """
    try:
        logging.info(f"Data loading started from: {data_dir}")
        loader = SimpleDirectoryReader(data_dir)  
        documents = loader.load_data()
        logging.info("Data loading completed.")
        return documents
    except Exception as e:
        logging.error("Exception during data loading.")
        raise customexception(e, sys)
