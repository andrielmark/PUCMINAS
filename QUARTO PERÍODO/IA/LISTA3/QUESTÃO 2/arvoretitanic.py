# Importando as bibliotecas para a Árvore de Decisão
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import pickle

# Carregando os dados pré-processados
with open('titanic_prepared.pkl', 'rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

# Criando e treinando o modelo de Árvore de Decisão
# Usamos 'entropy' como critério, pois ajuda a criar uma árvore mais balanceada
model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
predictions = model.predict(X_test)

# Avaliando o desempenho do modelo
print("Relatório de Classificação:")
print(classification_report(y_test, predictions))
print(f"Acurácia do modelo: {accuracy_score(y_test, predictions):.2f}")

# 1ª Figura: Matriz de Confusão
cm_display = ConfusionMatrixDisplay.from_estimator(
    model, X_test, y_test, cmap=plt.cm.Blues
)
cm_display.figure_.suptitle("Matriz de Confusão")
plt.show()

# 2ª Figura: Visualização da Árvore de Decisão
plt.figure(figsize=(20, 15))
plot_tree(
    model, 
    feature_names=X_train.columns, 
    class_names=['Não Sobreviveu', 'Sobreviveu'], 
    filled=True, 
    rounded=True, 
    fontsize=10
)
plt.title("Árvore de Decisão do Titanic", fontsize=18)
plt.show()

# Extraindo as regras da árvore
from sklearn.tree import export_text
tree_rules = export_text(model, feature_names=list(X_train.columns))
print("Regras da Árvore de Decisão:")
print(tree_rules)