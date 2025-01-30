# Filtro de Spam para E-mails

## Objetivo
O objetivo deste trabalho é desenvolver um filtro de spam eficiente que classifique automaticamente e-mails como "spam" ou "não spam". A solução será baseada em técnicas de aprendizado de máquina e processamento de linguagem natural (NLP). Espera-se que o modelo seja capaz de identificar padrões textuais comuns em mensagens de spam, garantindo alta acurácia na classificação.

## Metodologia

### Coleta de Dados
- Utilizaremos datasets públicos amplamente utilizados para estudos relacionados, como o **SpamAssassin Dataset** ou o **Enron Dataset**.

### Pré-Processamento
- **Limpeza dos textos**: remoção de HTML, URLs, e-mails, números e caracteres especiais.
- **Tokenização e remoção de palavras irrelevantes (stopwords)**.
- **Representação dos textos**: transformação para vetores numéricos usando o método **TF-IDF** (Term Frequency-Inverse Document Frequency).

### Treinamento do Modelo
- O modelo base será um classificador probabilístico como o **Naive Bayes**, conhecido pela eficiência em tarefas de classificação de textos.
- Para comparar desempenho, testaremos também outros algoritmos simples como **SVM** (Support Vector Machine).

### Validação e Teste
- Os dados serão divididos em conjuntos de treino (80%) e teste (20%).
- O desempenho será avaliado utilizando métricas como **acurácia**, **precisão**, **recall**, e **AUC-ROC**.

### Resultados e Análise
- Comparação dos resultados experimentais entre os modelos treinados.
- Discussão sobre a eficiência do modelo escolhido e as principais dificuldades encontradas durante a implementação.

## Justificativa
O envio de spam representa um problema significativo em termos de segurança cibernética e produtividade. Um filtro de spam eficiente pode auxiliar no gerenciamento de e-mails, evitando riscos e melhorando a experiência dos usuários. Este projeto explora técnicas simples, mas eficazes, que podem ser aplicadas em diversas áreas da computação.
