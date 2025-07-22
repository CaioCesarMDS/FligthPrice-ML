<h1>FlightPrice - Machine Learning</h1>

<h2>Discentes: Caio Cesar, Rodrigo Sena e Vitor Otávio</h2>

<br>

<div>
  <h2>📊 Sobre o Projeto</h2>
  <p>
    Este é um projeto de aprendizado de máquina com o objetivo de identificar, entre seis modelos de regressão, aquele que apresenta o melhor desempenho na predição de preços.
  </p>
  
  <p>
    Os modelos testados foram:
    <strong>LinearRegression</strong>, <strong>KNeighborsRegressor</strong>, <strong>DecisionTreeRegressor</strong>, <strong>RandomForestRegressor</strong>, <strong>GradientBoostingRegressor</strong> e <strong>HistGradientBoostingRegressor</strong>.
  </p>
  
  <p>
    Foram avaliadas diversas combinações de hiperparâmetros utilizando <strong>RandomizedSearchCV</strong> para otimizar o desempenho de cada modelo.
  </p>
  
  <p>
    O <strong>MLflow</strong> foi utilizado para registrar todos os experimentos e resultados, permitindo a visualização de cada execução, dos hiperparâmetros utilizados e das métricas de avaliação. Isso garante rastreabilidade e controle total do processo de modelagem.
  </p>
  
  <p>
    As métricas utilizadas para avaliação dos modelos foram:
    <strong>Root Mean Squared Error (RMSE)</strong>, <strong>Mean Absolute Error (MAE)</strong> e <strong>R² Score (R²)</strong>.
  </p>

  <p>
    O projeto foi conteinerizado com <strong>Docker</strong>, utilizando dois containers: um para o servidor MLflow e outro para a execução do pipeline de treinamento.
  </p>

  <p>
    Após o treinamento e a seleção do melhor modelo, é realizada a predição sobre o conjunto de teste, gerando o arquivo <code>test_set_with_predictions.xlsx</code> na pasta <code>data/</code>, com os preços previstos.
  </p>
</div>

<br>

<h1>Instalação / Execução</h1>
<p>Para executar o projeto localmente, siga estas etapas:</p>

1 - Clone esse repositório em sua máquina local:

```
git clone https://github.com/CaioCesarMDS/FligthPrice-ML.git
```

2 - Suba o container com docker:

```
docker compose up --build
```

3 - Espere o treinamento do modelo (+ / - 12m): 


4 - Abra o servidor do Mlflow no navegador para ver todo os experimentos:

```
http:localhost:5000
```

<br>

<h1>Principais bibliotecas e ferramentas Utilizadas</h1>
<ul>
  <li>Pandas</li>
  <li>Scikit_Learn</li>
  <li>Numpy</li>
  <li>Mlflow</li>
</ul>
