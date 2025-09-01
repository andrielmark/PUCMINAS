import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import pickle

# Carregar dados
with open('restaurante.pkl', 'rb') as f:
    X_treino, X_teste, y_treino, y_teste = pickle.load(f)

# Criar e treinar modelo
modelo = DecisionTreeClassifier(criterion='entropy', random_state=42)
modelo.fit(X_treino, y_treino)

# Previsões
previsoes = modelo.predict(X_teste)

# Avaliação
print(f"Accuracy: {accuracy_score(y_teste, previsoes):.2f}")
print(classification_report(y_teste, previsoes))

# 1ª Figura: Matriz de Confusão
cm_display = ConfusionMatrixDisplay.from_estimator(
    modelo, X_teste, y_teste, cmap=plt.cm.Blues
)
cm_display.figure_.suptitle("Matriz de Confusão")
cm_display.figure_.tight_layout()  # melhor espaçamento
plt.show()  # exibe a matriz

# 2ª Figura: Árvore de Decisão
plt.figure(figsize=(15,15))
plot_tree(
    modelo, 
    feature_names=X_treino.columns, 
    class_names=modelo.classes_, 
    filled=True, 
    rounded=True, 
    fontsize=10
)
plt.title("Árvore de Decisão", fontsize=16)
plt.tight_layout()  # organiza melhor a árvore
plt.show()  # exibe a árvore
