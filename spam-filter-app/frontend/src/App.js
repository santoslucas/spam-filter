import React, { useEffect, useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [emails, setEmails] = useState([]);

  useEffect(() => {
    // Fetch emails from the backend
    axios.get('http://127.0.0.1:5000/emails')
      .then(response => {
        setEmails(response.data);
      })
      .catch(error => {
        console.error('There was an error fetching the emails!', error);
      });
      console.log("email", emails);
  }, []);

  return (
    <div className="App">
      <h1>Spam Detector</h1>
      <table>
        <thead>
          <tr>
            <th>Label</th>
            <th>Text</th>
          </tr>
        </thead>
        <tbody>
          {emails.map((email, index) => (
            <tr key={index} className={email.label === 'spam' ? 'spam' : 'ham'}>
              <td>{email.label}</td>
              <td>{email.text}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default App;