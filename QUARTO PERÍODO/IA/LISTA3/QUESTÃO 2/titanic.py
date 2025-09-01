# Importando as bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Carregando os dados
df = pd.read_csv('train.csv')

# Visualizando as primeiras linhas do DataFrame e informações gerais
print(df.head())
print("\nInformações sobre os dados:")
df.info()

# Verificando a distribuição da variável 'Survived' (Sobreviventes)
print("\nContagem de sobreviventes e não-sobreviventes:")
print(df['Survived'].value_counts())
sns.countplot(x='Survived', data=df)
plt.title('Distribuição de Sobreviventes (0 = Não, 1 = Sim)')
plt.show()

# Distribuição da variável 'Sex' (Gênero)
print("\nContagem de passageiros por gênero:")
print(df['Sex'].value_counts())
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title('Sobrevivência por Gênero')
plt.show()

# Distribuição da variável 'Pclass' (Classe do bilhete)
print("\nContagem de passageiros por classe:")
print(df['Pclass'].value_counts())
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Sobrevivência por Classe do Bilhete')
plt.show()

# Análise de 'Age' (Idade)
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'].dropna(), bins=30, kde=True)
plt.title('Distribuição da Idade')
plt.show()

# Sobrevivência por idade (gráfico de densidade)
plt.figure(figsize=(10, 6))
sns.kdeplot(df.loc[df['Survived'] == 0, 'Age'].dropna(), label='Não Sobreviveu')
sns.kdeplot(df.loc[df['Survived'] == 1, 'Age'].dropna(), label='Sobreviveu')
plt.title('Densidade da Idade por Sobrevivência')
plt.legend()
plt.show()

# Análise de 'Fare' (Tarifa)
plt.figure(figsize=(10, 6))
sns.histplot(df['Fare'], bins=30, kde=True)
plt.title('Distribuição da Tarifa')
plt.show()

# Tratar valores ausentes
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop('Cabin', axis=1, inplace=True) # A coluna 'Cabin' tem muitos valores ausentes, o que a torna difícil de usar

# Codificar atributos categóricos
# Os atributos 'Sex' e 'Embarked' são categóricos e precisam ser transformados para que o modelo possa entendê-los.
# Para 'Sex', usaremos Label Encoding. Para 'Embarked', que tem mais de duas categorias, usaremos One-Hot Encoding.
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Separando features (X) e target (y)
X = df.drop(['Survived', 'Name', 'Ticket'], axis=1) # Remove colunas não relevantes para o modelo
y = df['Survived']

# Aplicando One-Hot Encoding em 'Embarked'
X = pd.get_dummies(X, columns=['Embarked'], drop_first=True)

# Divisão dos dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Salvando os dados pré-processados para uso futuro
import pickle
with open('titanic_prepared.pkl', 'wb') as f:
    pickle.dump([X_train, X_test, y_train, y_test], f)

print("\nDados pré-processados e salvos em 'titanic_prepared.pkl'")