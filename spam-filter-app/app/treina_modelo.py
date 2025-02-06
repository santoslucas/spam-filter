import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import requests
from io import BytesIO
from zipfile import ZipFile
import joblib
from sklearn.model_selection import train_test_split

def carregar_e_salvar_modelo():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    response = requests.get(url)

    with ZipFile(BytesIO(response.content)) as zip_file:
        with zip_file.open("SMSSpamCollection") as file:
            df = pd.read_csv(file, sep='\t', header=None, names=['label', 'text'])

    df['label'] = df['label'].map({'spam': 1, 'ham': 0})

    df['text'] = df['text'].str.replace(r'[^\w\s]', '', regex=True)
    df['text'] = df['text'].str.lower()

    df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)

    tfidf = TfidfVectorizer(stop_words='english')
    X_train = tfidf.fit_transform(df_train['text'])

    svm_model = SVC(kernel='linear', probability=True, random_state=42)
    svm_model.fit(X_train, df_train['label'])

    joblib.dump(tfidf, 'tfidf_vectorizer.joblib')
    joblib.dump(svm_model, 'svm_model.joblib')
    df_test.to_csv('dados_teste.csv', index=False)

carregar_e_salvar_modelo()