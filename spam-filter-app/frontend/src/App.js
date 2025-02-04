import React, { useEffect, useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [emails, setEmails] = useState([]);

  // Carrega os e-mails ao iniciar
  useEffect(() => {
    axios.get('http://127.0.0.1:5000/emails')
      .then(response => {
        setEmails(response.data);
      })
      .catch(error => {
        console.error("Erro ao carregar e-mails:", error);
      });
  }, []);

  // Função para verificar se um e-mail é spam
  const handleCheckSpam = async (emailId, emailText, actualLabel) => {
    try {
      const response = await axios.post('http://127.0.0.1:5000/classificar', {
        corpo_email: emailText, // Corrigido para enviar o texto do e-mail
      });
      const isSpam = response.data.resultado; // Corrigido para acessar o campo correto
      console.log("label" , actualLabel);
      console.log("isSpam" , isSpam);
      // Verifica se a previsão foi correta
      const isCorrect = (isSpam && actualLabel === 1) || (!isSpam && actualLabel === 0);

      // Atualiza apenas o e-mail específico
      setEmails(prevEmails =>
        prevEmails.map(email =>
          email.id === emailId
            ? { ...email, is_spam: isSpam, is_correct: isCorrect, prediction: isSpam ? 'Spam' : 'Não é Spam' }
            : email
        )
      );
    } catch (error) {
      console.error("Erro ao verificar spam:", error);
    }
  };

  return (
    <div className="App">
      <header className="header">
        <h1>Detector de Spam</h1>
      </header>
      <div className="content">
        <table>
          <thead>
            <tr>
              <th>Spam</th>
              <th>E-mail</th>
              <th>Ação</th>
              <th>Previsão</th>
              <th>Acerto</th>
            </tr>
          </thead>
          <tbody>
            {emails.map(email => (
              <tr key={email.id}>
                <td>{email.id}</td>
                <td>{email.text}</td>
                <td>
                  <button onClick={() => handleCheckSpam(email.id, email.text, email.label)}>
                    Verificar Spam
                  </button>
                </td>
                <td>
                  {email.prediction && (
                    <span className={email.prediction === 'Spam' ? 'spam' : 'ham'}>
                      {email.prediction}
                    </span>
                  )}
                </td>
                <td>
                  {email.is_correct !== undefined && (
                    <span className={email.is_correct ? 'correct' : 'incorrect'}>
                      {email.is_correct ? '✔️ Correto' : '❌ Errado'}
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default App;