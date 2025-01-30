import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import requests
from io import BytesIO
from zipfile import ZipFile
import joblib  # Biblioteca para salvar/carregar modelos

# Função para carregar e preparar o modelo
def carregar_e_salvar_modelo():
    # Baixando o dataset SMS Spam Collection
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    response = requests.get(url)

    # Extraindo o arquivo ZIP
    with ZipFile(BytesIO(response.content)) as zip_file:
        with zip_file.open("SMSSpamCollection") as file:
            df = pd.read_csv(file, sep='\t', header=None, names=['label', 'text'])

    # Convertendo as labels para spam (1) e ham (0)
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})

    # Limpeza dos textos: remoção de caracteres especiais e conversão para minúsculas
    df['text'] = df['text'].str.replace(r'[^\w\s]', '', regex=True)
    df['text'] = df['text'].str.lower()

    # Tokenização e remoção de stopwords
    tfidf = TfidfVectorizer(stop_words='english')
    X = tfidf.fit_transform(df['text'])

    # Treinamento do modelo Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X, df['label'])

    # Salvar o modelo e o vetorizador em arquivos
    joblib.dump(tfidf, 'tfidf_vectorizer.joblib')
    joblib.dump(nb_model, 'naive_bayes_model.joblib')

# Executar a função para salvar o modelo e o vetorizador
carregar_e_salvar_modelo()