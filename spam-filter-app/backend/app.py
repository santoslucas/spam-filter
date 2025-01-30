from flask import Flask, jsonify
import pandas as pd
import zipfile
from io import BytesIO
import requests


from flask_cors import CORS

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
    emails = df.to_dict(orient='records')
    print(emails[0])
    return jsonify(emails)

if __name__ == '__main__':
    app.run(debug=True)

