# Document Processor and Question Answering System

This Python script provides functionalities for processing documents, embedding text, and performing question answering tasks using Hugging Face models. Below is a guide on how to use the script along with necessary setup instructions.

## Setup

1. **Installation**:

   - Clone or download the `langchain` package from the official repository.

2. **Dependencies**:

   - Ensure you have the following dependencies installed:
     - `textwrap`
     - `os`
     - `langchain` package (including `langchain.document_loaders`, `langchain.text_splitter`, `langchain.embeddings`, `langchain.vectorstores`, and `langchain.chains.question_answering` modules)
     - `HuggingFaceHub`

3. **Data File**:

   - Ensure you have a text file named `data.txt` containing the input text data for processing.

## Usage

- Define the file path and query.
- Execute the script to process the document, embed text, and perform question answering.

## Explanation

- The `DocumentProcessor` class loads, preprocesses, and chunks the input text document.
- The `TextEmbedder` class embeds the chunked documents using Hugging Face embeddings and FAISS indexing.
- The `QuestionAnsweringSystem` class handles loading a question-answering chain and answering user queries using Hugging Face models.
- Ensure the `data.txt` file is present in the same directory as the script for proper execution.
