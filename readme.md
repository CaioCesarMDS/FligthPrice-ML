<h1>✈️ FlightPrice - Machine Learning</h1>

<h2>Discentes: Caio Cesar, Rodrigo Sena e Vitor Otávio</h2>

<br>

<div>
  <h2>📊 Sobre o Projeto</h2>

  <p>
    Este é um projeto de aprendizado de máquina cujo objetivo é identificar, entre seis modelos de regressão, aquele que apresenta o melhor desempenho na predição de preços de passagens aéreas.
  </p>
  
  <p>
    Os modelos avaliados foram:
    <strong>LinearRegression</strong>, <strong>KNeighborsRegressor</strong>, <strong>DecisionTreeRegressor</strong>, <strong>RandomForestRegressor</strong>, <strong>GradientBoostingRegressor</strong> e <strong>HistGradientBoostingRegressor</strong>.
  </p>
  
  <p>
    Diversas combinações de hiperparâmetros foram testadas com o <strong>RandomizedSearchCV</strong>, visando a melhor performance possível para cada modelo.
  </p>
  
  <p>
    Utilizamos o <strong>MLflow</strong> para rastrear os experimentos, armazenar os resultados e permitir a análise detalhada de cada execução — incluindo hiperparâmetros utilizados e métricas de avaliação.
  </p>
  
  <p>
    As métricas de desempenho aplicadas foram:
    <strong>Root Mean Squared Error (RMSE)</strong>, <strong>Mean Absolute Error (MAE)</strong> e <strong>R² Score</strong>.
  </p>

  <p>
    O projeto foi conteinerizado com <strong>Docker</strong>, utilizando dois containers:
    um para o servidor MLflow e outro para o script de treinamento e predição.
  </p>

  <p>
    Após o treinamento e escolha do melhor modelo, é realizada a predição sobre os dados de teste, gerando o arquivo <code>test_set_with_predictions.xlsx</code> na pasta <code>data/</code>.
  </p>
</div>

<br>

<div>
  <h2>📁 Estrutura do Projeto</h2>

  <pre>
  .
  ├── data/
  │   ├── test_set.xlsx           # Dados de teste utilizados para predição final
  │   └── train_set.xlsx          # Dados de treino utilizados para treinar os modelos
  │
  ├── docker-compose.yaml         # Arquivo de orquestração Docker (MLflow + treinamento)
  ├── Dockerfile                  # Define a imagem do container para rodar o treinamento
  ├── readme.md                   # Documentação do projeto
  ├── requirements.txt            # Dependências do Python
  │
  └── src/                        # Código-fonte principal
      ├── main.py                 # Script principal que orquestra o fluxo de execução
      │
      ├── models/
      │   └── model_config.py     # Define os modelos e seus hiperparâmetros
      │
      ├── training/
      │   └── train_models.py     # Lógica de treino, avaliação e logging via MLflow
      │
      └── utils/                  # Funções auxiliares
          ├── helpers.py          # Funções utilitárias
          ├── predict_and_save.py # Predição com dados de teste e salvamento de resultados na pasta data
          └── preprocessing.py    # Pré-processamento dos dados
  </pre>

</div>

<br>

<div>
  <h2>🏆 Melhores Resultados de Cada Modelo</h2>

  <table>
  <thead>
    <tr>
      <th>Modelo</th>
      <th>R² Score</th>
      <th>RMSE</th>
      <th>MAE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>LinearRegression</td>
      <td>0.699</td>
      <td>2221.78</td>
      <td>1666.39</td>
    </tr>
    <tr>
      <td>KNeighborsRegressor</td>
      <td>0.848</td>
      <td>1575.79</td>
      <td>855.22</td>
    </tr>
    <tr>
      <td>DecisionTreeRegressor</td>
      <td>0.878</td>
      <td>1412.24</td>
      <td>779.23</td>
    </tr>
    <tr>
      <td>RandomForestRegressor</td>
      <td>0.917</td>
      <td>1165.91</td>
      <td>613.21</td>
    </tr>
    <tr>
      <td>GradientBoostingRegressor</td>
      <td>0.929</td>
      <td>1081.43</td>
      <td>624.73</td>
    </tr>
    <tr>
      <td>HistGradientBoostingRegressor</td>
      <td>0.933</td>
      <td>1045.05</td>
      <td>593.62</td>
    </tr>
  </tbody>
</table>
</div>

<br>

<div>
  <h2>🛠️ Instalação e Execução</h2>

  <p>Siga os passos abaixo para executar o projeto localmente:</p>

  1 - Clone esse repositório em sua máquina local:
  
  ```
  git clone https://github.com/CaioCesarMDS/FligthPrice-ML.git
  ```
  
  2 - Suba o container com docker:
  
  ```
  docker compose up --build
  ```
  
  3 - Espere o treinamento do modelo (+ / - 12m): 

  * É possível diminuir o tempo necessário alterando o "n_iter" e "cv" da RandomizedSearchCV para baixo, com a consequência da piora dos resultados.
  
  
  4 - Abra o servidor do Mlflow no navegador para ver todo os experimentos:
  
  ```
  http:localhost:5000
  ```    

</div>

<br>

<div>
  <h2>📚 Principais Bibliotecas e Ferramentas Utilizadas</h2>

  <ul>
    <li>Pandas</li>
    <li>NumPy</li>
    <li>Scikit-learn</li>
    <li>MLflow</li>
    <li>Docker</li>
  </ul>
</div>
