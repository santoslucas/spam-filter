from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Carregar o modelo e o vetorizador salvos
tfidf = joblib.load('tfidf_vectorizer.joblib')
nb_model = joblib.load('naive_bayes_model.joblib')

# Criar a aplicação Flask
app = Flask(__name__)

# Rota para classificar o e-mail
@app.route('/classificar', methods=['POST'])
def classificar_email():
    # Receber o corpo do e-mail do JSON enviado pelo JavaScript
    data = request.json
    corpo_email = data.get('corpo_email', '')

    # Pré-processamento do texto do e-mail
    corpo_email = pd.Series(corpo_email)
    corpo_email = corpo_email.str.replace(r'[^\w\s]', '', regex=True)
    corpo_email = corpo_email.str.lower()

    # Transformar o texto em uma matriz TF-IDF
    email_tfidf = tfidf.transform(corpo_email)

    # Fazer a predição
    predicao = nb_model.predict(email_tfidf)

    # Retornar o resultado como JSON
    resultado = True if predicao[0] == 1 else False
    return jsonify({"resultado": resultado})

# Iniciar o servidor Flask
if __name__ == '__main__':
    app.run(debug=True)