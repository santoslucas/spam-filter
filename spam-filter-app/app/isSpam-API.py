from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

tfidf = joblib.load('tfidf_vectorizer.joblib')
svm_model = joblib.load('svm_model.joblib')

df_test = pd.read_csv('dados_teste.csv')

app = Flask(__name__)
CORS(app) 

@app.route('/emails', methods=['GET'])
def get_emails():
    emails = df_test.to_dict(orient='records')
    for i, email in enumerate(emails):
        email['id'] = i + 1
    return jsonify(emails)

@app.route('/classificar', methods=['POST'])
def classificar_email():
    data = request.json
    corpo_email = data.get('corpo_email', '')

    corpo_email = corpo_email.lower()
    corpo_email = ''.join([char for char in corpo_email if char.isalnum() or char == ' '])
    
    email_tfidf = tfidf.transform([corpo_email])

    predicao = svm_model.predict(email_tfidf)

    resultado = bool(predicao[0])
    return jsonify({"resultado": resultado})

if __name__ == '__main__':
    app.run(debug=True)