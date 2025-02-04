import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC  # Importando o SVM
import requests
from io import BytesIO
from zipfile import ZipFile
import joblib
from sklearn.model_selection import train_test_split

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

    # Dividindo o dataset em treino (70%) e teste (30%)
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)

    # Tokenização e remoção de stopwords (apenas nos dados de treino)
    tfidf = TfidfVectorizer(stop_words='english')
    X_train = tfidf.fit_transform(df_train['text'])

    # Treinamento do modelo SVM
    svm_model = SVC(kernel='linear', probability=True, random_state=42)  # Usando kernel linear
    svm_model.fit(X_train, df_train['label'])

    # Salvar o modelo, o vetorizador e os dados de teste
    joblib.dump(tfidf, 'tfidf_vectorizer.joblib')
    joblib.dump(svm_model, 'svm_model.joblib')  # Salvando o modelo SVM
    df_test.to_csv('dados_teste.csv', index=False)  # Salvar dados de teste em um arquivo CSV

# Executar a função para salvar o modelo e os dados de teste
carregar_e_salvar_modelo()