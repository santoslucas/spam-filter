# Importação das bibliotecas necessárias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix
import requests
from io import BytesIO
from zipfile import ZipFile

# 1. Coleta de Dados
# Baixando o dataset SMS Spam Collection
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
response = requests.get(url)

# Extraindo o arquivo ZIP
with ZipFile(BytesIO(response.content)) as zip_file:
    with zip_file.open("SMSSpamCollection") as file:
        df = pd.read_csv(file, sep='\t', header=None, names=['label', 'text'])

# Exibindo a distribuição das classes
print("\nDistribuição das classes:")
print(df['label'].value_counts())

# 2. Pré-Processamento
# Convertendo as labels para spam (1) e ham (0)
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# Limpeza dos textos: remoção de caracteres especiais e conversão para minúsculas
df['text'] = df['text'].str.replace(r'[^\w\s]', '', regex=True)
df['text'] = df['text'].str.lower()

# Tokenização e remoção de stopwords
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(df['text'])

# 3. Treinamento do Modelo
# Dividindo os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.2, random_state=42)

# Modelo Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Modelo SVM
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

# 4. Validação e Teste
# Avaliação do modelo Naive Bayes
y_pred_nb = nb_model.predict(X_test)
print("\nNaive Bayes Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Precision:", precision_score(y_test, y_pred_nb, zero_division=0))
print("Recall:", recall_score(y_test, y_pred_nb, zero_division=0))
print("AUC-ROC:", roc_auc_score(y_test, nb_model.predict_proba(X_test)[:, 1]))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_nb))
print("Classification Report:\n", classification_report(y_test, y_pred_nb))

# Avaliação do modelo SVM
y_pred_svm = svm_model.predict(X_test)
print("\nSVM Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Precision:", precision_score(y_test, y_pred_svm, zero_division=0))
print("Recall:", recall_score(y_test, y_pred_svm, zero_division=0))
print("AUC-ROC:", roc_auc_score(y_test, svm_model.predict_proba(X_test)[:, 1]))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))

# 5. Validação Cruzada (Opcional)
# Usando validação cruzada com 5 folds para o modelo Naive Bayes
scores_nb = cross_val_score(nb_model, X, df['label'], cv=5, scoring='accuracy')
print("\nValidação Cruzada (Naive Bayes):")
print("Acurácia média com validação cruzada (5 folds):", scores_nb.mean())

# Usando validação cruzada com 5 folds para o modelo SVM
scores_svm = cross_val_score(svm_model, X, df['label'], cv=5, scoring='accuracy')
print("\nValidação Cruzada (SVM):")
vx