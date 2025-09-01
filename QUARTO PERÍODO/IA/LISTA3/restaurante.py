import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
base = pd.read_csv('restaurante.csv', sep=';')
import pandas as pd
# Lendo com o separador correto
base = pd.read_csv("restaurante.csv", sep=";")
# Transformando a coluna 'Cliente'
base["Cliente"] = base["Cliente"].map({
    "nenhum": 0,
    "alguns": 1,
    "cheio": 2,
    "Nenhum": 0,
    "Alguns": 1,
    "Cheio": 2
})
# Conferindo
print(base.head())
Classificação = base.columns[-1]
np.unique(base[Classificação], return_counts=True)
sns.countplot(x = base[Classificação]);
from sklearn.preprocessing import LabelEncoder
#para codificar todos os atributos para laberEncoder de uma única vez
#base_encoded = base.apply(LabelEncoder().fit_transform)
cols_label_encode = ['Alternativo', 'Bar', 'SexSab','fome','Preco', 'Chuva', 'Res','Tempo']
base[cols_label_encode] = base[cols_label_encode].apply(LabelEncoder().fit_transform)
len(np.unique(base['Cliente']))
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
cols_onehot_encode = ['Tipo']
# Inicializar o OneHotEncoder (sparse_output=False retorna um array denso)
onehot = OneHotEncoder(sparse_output=False)
# Aplicar o OneHotEncoder apenas nas colunas categóricas
df_onehot = onehot.fit_transform(base[cols_onehot_encode])
# Obter os novos nomes das colunas após a codificação
nomes_das_colunas = onehot.get_feature_names_out(cols_onehot_encode)
# Criar um DataFrame com os dados codificados e as novas colunas
df_onehot = pd.DataFrame(df_onehot, columns=nomes_das_colunas)
# Combinar as colunas codificadas com as colunas que não foram transformadas
base_encoded= pd.concat([df_onehot, base.drop(columns=cols_onehot_encode)], axis=1)
# Supondo que a última coluna seja o target
X_prev= base_encoded.iloc[:, :-1]
y_classe = base_encoded.iloc[:, -1]
from sklearn.model_selection import train_test_split
#X_train_ds, X_test_ds, y_train_ds, y_test_ds = train_test_split(X, y, test_size=0.3, random_state=123, shuffle=True, stratify=y)
X_treino, X_teste, y_treino, y_teste = train_test_split(X_prev, y_classe, test_size = 0.20, random_state = 42)
import pickle
with open('Restaurante.pkl', mode = 'wb') as f:
  pickle.dump([X_treino, X_teste, y_treino, y_teste], f)
