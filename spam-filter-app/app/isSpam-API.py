from flask import Flask, request, jsonify
import joblib
import pandas as pd
import zipfile
from io import BytesIO
import requests
from flask_cors import CORS

# Carregar o modelo e o vetorizador salvos
tfidf = joblib.load('tfidf_vectorizer.joblib')
nb_model = joblib.load('naive_bayes_model.joblib')

# Criar a aplicação Flask
app = Flask(__name__)
CORS(app)  # Habilita CORS para todas as rotas

# Carregar a base de dados
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
response = requests.get(url)
with zipfile.ZipFile(BytesIO(response.content)) as thezip:
    with thezip.open('SMSSpamCollection') as thefile:
        df = pd.read_csv(thefile, sep='\t', header=None, names=['label', 'text'])

# Rota para obter os e-mails
@app.route('/emails', methods=['GET'])
def get_emails():
    # Adicionar um ID único para cada e-mail
    emails = df.to_dict(orient='records')
    for i, email in enumerate(emails):
        email['id'] = i + 1  # Adiciona um ID único
    return jsonify(emails)

# Rota para classificar o e-mail
@app.route('/classificar', methods=['POST'])
def classificar_email():
    # Receber o corpo do e-mail do JSON enviado pelo JavaScript
    data = request.json
    corpo_email = data.get('corpo_email', '')

    # Pré-processamento do texto do e-mail
    corpo_email = corpo_email.lower()  # Converter para minúsculas
    corpo_email = ''.join([char for char in corpo_email if char.isalnum() or char == ' '])  # Remover caracteres especiais

    # Transformar o texto em uma matriz TF-IDF
    email_tfidf = tfidf.transform([corpo_email])

    # Fazer a predição
    predicao = nb_model.predict(email_tfidf)

    # Retornar o resultado como JSON
    resultado = bool(predicao[0])  # Converter para booleano
    print(resultado);
    return jsonify({"resultado": resultado})

# Iniciar o servidor Flask
if __name__ == '__main__':
    app.run(debug=True)