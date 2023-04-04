import urllib.request
import fitz
import re
import numpy as np
import tensorflow_hub as hub
import openai
import os
from sklearn.neighbors import NearestNeighbors
import streamlit as st

def download_pdf(url, output_path):
    urllib.request.urlretrieve(url, output_path)


def preprocess(text):
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    return text


def pdf_to_text(path, start_page=1, end_page=None):
    doc = fitz.open(path)
    total_pages = doc.page_count

    if end_page is None:
        end_page = total_pages

    text_list = []

    for i in range(start_page-1, end_page):
        text = doc.load_page(i).get_text("text")
        text = preprocess(text)
        text_list.append(text)

    doc.close()
    return text_list


def text_to_chunks(texts, word_length=150, start_page=1):
    text_toks = [t.split(' ') for t in texts]
    page_nums = []
    chunks = []
    
    for idx, words in enumerate(text_toks):
        for i in range(0, len(words), word_length):
            chunk = words[i:i+word_length]
            if (i+word_length) > len(words) and (len(chunk) < word_length) and (
                len(text_toks) != (idx+1)):
                text_toks[idx+1] = chunk + text_toks[idx+1]
                continue
            chunk = ' '.join(chunk).strip()
            chunk = f'[{idx+start_page}]' + ' ' + '"' + chunk + '"'
            chunks.append(chunk)
    return chunks


class SemanticSearch:
    
    def __init__(self):
        self.use = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
        self.fitted = False
    
    
    def fit(self, data, batch=1000, n_neighbors=5):
        self.data = data
        self.embeddings = self.get_text_embedding(data, batch=batch)
        n_neighbors = min(n_neighbors, len(self.embeddings))
        self.nn = NearestNeighbors(n_neighbors=n_neighbors)
        self.nn.fit(self.embeddings)
        self.fitted = True
    
    
    def __call__(self, text, return_data=True):
        inp_emb = self.use([text])
        neighbors = self.nn.kneighbors(inp_emb, return_distance=False)[0]
        
        if return_data:
            return [self.data[i] for i in neighbors]
        else:
            return neighbors
    
    
    def get_text_embedding(self, texts, batch=1000):
        embeddings = []
        for i in range(0, len(texts), batch):
            text_batch = texts[i:(i+batch)]
            emb_batch = self.use(text_batch)
            embeddings.append(emb_batch)
        embeddings = np.vstack(embeddings)
        return embeddings



# The modified function generates embeddings based on PDF file name and page number and checks if the embeddings file exists before loading or generating it.	
def load_recommender(path, start_page=1):
    global recommender
    pdf_file = os.path.basename(path)
    embeddings_file = f"{pdf_file}_{start_page}.npy"
# Crear elementos de entrada en la interfaz de Streamlit
url = st.text_input("URL")
file = st.file_uploader("Subir archivo")
question = st.text_input("Pregunta")
openAI_key = st.text_input("Clave de API de OpenAI", type="password")

# Crear un botón para activar la función question_answer
if st.button("Obtener respuesta"):
    answer = question_answer(url, file, question, openAI_key)
    st.write(answer)

# Definir la función question_answer
def question_answer(url, file, question, openAI_key):
    # Aquí va la lógica para procesar la pregunta y obtener la respuesta
    # utilizando la API de OpenAI
    pass

# Lanzar la aplicación de Streamlit
st.title("Demo de Streamlit y OpenAI")
    
   
