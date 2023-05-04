from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv


import pandas as pd
import tiktoken

from tqdm import tqdm

import os
import zipfile
from urllib.request import urlopen
from io import BytesIO
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def embed_document(vector_db, splitter, document_id, document):
    metadata = [{'document_id': document_id}]
    split_documents = splitter.create_documents([str(document)], metadatas=metadata)

    texts = [d.page_content for d in split_documents]
    metadatas = [d.metadata for d in split_documents]

    docsearch = vector_db.add_texts(texts, metadatas=metadatas)

def zipfile_from_github():
    http_response = urlopen('https://github.com/hango100/gpt4-pdf-chatbot-langchain/archive/refs/heads/main.zip')
    zf = BytesIO(http_response.read())
    return zipfile.ZipFile(zf, 'r')

embeddings = OpenAIEmbeddings(
    openai_api_key='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',
    # openai_organization=os.environ['OPENAI_ORG_ID'],
)
encoder = tiktoken.get_encoding('cl100k_base')

pinecone.init(
    api_key='ad1f4867-3f72-4953-8d70-8f7c986268d1',
    environment='us-west1-gcp'
)
vector_store = Pinecone(
    index=pinecone.Index('codebase'),
    embedding_function=embeddings.embed_query,
    text_key='text',
    namespace='code'
)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

total_tokens, corpus_summary = 0, []
file_texts, metadatas = [], []
with zipfile_from_github() as zip_ref:
    zip_file_list = zip_ref.namelist()
    
    pbar = tqdm(zip_file_list, desc=f'Total tokens: 0')
    for file_name in pbar:
        if (file_name.endswith('/') or 
            any(f in file_name for f in ['.DS_Store', '.gitignore']) or 
            any(file_name.endswith(ext) for ext in ['.png', '.jpg', '.jpeg'])
        ):
            continue
        else:
            with zip_ref.open(file_name, 'r') as file:
                file_contents = str(file.read())
                file_name_trunc = str(file_name).replace('the-algorithm-main/', '')
                
                n_tokens = len(encoder.encode(file_contents))
                total_tokens += n_tokens
                corpus_summary.append({'file_name': file_name_trunc, 'n_tokens': n_tokens})

                file_texts.append(file_contents)
                metadatas.append({'document_id': file_name_trunc})
                pbar.set_description(f'Total tokens: {total_tokens}')

split_documents = splitter.create_documents(file_texts, metadatas=metadatas)
vector_store.from_documents(
    documents=split_documents, 
    embedding=embeddings,
    index_name='codebase',
    namespace='code'
)

pd.DataFrame.from_records(corpus_summary).to_csv('data/corpus_summary.csv', index=False)