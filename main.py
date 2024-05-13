import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
scaler = StandardScaler()

df = pd.read_excel("logit_data_2.xlsx")
df.head()

# set X (training data) and y (target variable)
n_features = len(df.columns) - 1

X = df.iloc[:, :n_features].values
y = df.iloc[:, n_features].values.reshape(-1, 1)

mean = X.mean(axis=0)
std = X.std(axis=0)

scaler.fit(X)
X = scaler.transform(X)

def insert_ones(X):
    ones = np.ones((X.shape[0], 1))
    return np.concatenate((ones, X), axis=1)

# Função sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Carregando dados
df = pd.read_excel("logit_data_2.xlsx")
n_features = len(df.columns) - 1
X = df.iloc[:, :n_features].values
y = df.iloc[:, n_features].values.reshape(-1, 1)

# Normalização dos dados
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
X = insert_ones(X)

# Inicialização dos pesos aleatórios
w = np.random.rand(1, n_features + 1)

# Parâmetros do modelo
alpha = 0.01  # Taxa de aprendizado
epoch = 10000

# Treinamento do modelo
for _ in range(epoch):
    w = w - (alpha / len(X)) * np.sum((sigmoid(X @ w.T) - y) * X, axis=0)

# Impressão dos pesos (coeficientes)
# st.write("Intercepto:", w[0, 0])
# st.write("Coeficientes das variáveis independentes:", w[0, 1:])

# Interface para entrada de dados
st.write("# Previsão de churn para nova escola")
qnt_aluno = st.number_input("Quantidade de Alunos:")
tx_adesao = st.number_input("Taxa de Adesão:")
tx_leitura = st.number_input("Taxa de Leitura:")
qnt_msg_recebida = st.number_input("Quantidade de mensagem recebida:")
qnt_msg_enviada = st.number_input("Quantidade de mensagem enviada:")

# Exemplo de predição para uma nova escola
escola1 = np.array([[qnt_aluno, tx_adesao, tx_leitura, qnt_msg_recebida, qnt_msg_enviada]])
escola1 = scaler.transform(escola1)  # Aplicando a mesma escala que foi usada nos dados de treinamento
escola1 = insert_ones(escola1)
prediction = sigmoid(escola1 @ w.T)
st.write("Probabilidade de churn para a nova escola:", prediction)