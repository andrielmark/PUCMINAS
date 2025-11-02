# ============================================================
# Questão 1 – Problema do XOR
# Autor: Andriel Mark da Silva Pinto — Matrícula: 1528633
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# Entradas e saídas do problema XOR
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# Estrutura da rede
INPUT_LAYER_SIZE = 2
HIDDEN_LAYER_SIZE = 2
OUTPUT_LAYER_SIZE = 1
EPOCHS = 5000
LEARNING_RATE = 0.5

# Função de ativação Sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

# Inicialização dos pesos (He normalizada)
np.random.seed(42)
W1 = np.random.randn(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE) * np.sqrt(2. / INPUT_LAYER_SIZE)
B1 = np.zeros((1, HIDDEN_LAYER_SIZE))

W2 = np.random.randn(HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE) * np.sqrt(2. / HIDDEN_LAYER_SIZE)
B2 = np.zeros((1, OUTPUT_LAYER_SIZE))

# Vetor para armazenar o erro
errors = []

# ===============================
# Treinamento com Backpropagation
# ===============================
for epoch in range(EPOCHS):
    # Forward
    Z1 = np.dot(X, W1) + B1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + B2
    yHat = sigmoid(Z2)

    # Erro (MSE)
    error = y - yHat
    loss = np.mean(np.square(error))
    errors.append(loss)

    # Backpropagation
    dZ2 = error * sigmoid_derivative(yHat)
    dW2 = np.dot(A1.T, dZ2)
    dB2 = np.sum(dZ2, axis=0, keepdims=True)

    dZ1 = np.dot(dZ2, W2.T) * sigmoid_derivative(A1)
    dW1 = np.dot(X.T, dZ1)
    dB1 = np.sum(dZ1, axis=0, keepdims=True)

    # Atualização dos pesos
    W2 += LEARNING_RATE * dW2
    B2 += LEARNING_RATE * dB2
    W1 += LEARNING_RATE * dW1
    B1 += LEARNING_RATE * dB1

# ===============================
# Resultados
# ===============================
print("=== RESULTADOS FINAIS (XOR) ===")
for i in range(len(X)):
    print(f"Entrada: {X[i]} -> Saída prevista: {yHat[i][0]:.4f}")

# Gráfico da curva de erro
plt.plot(errors)
plt.title("Curva de Erro - XOR")
plt.xlabel("Época")
plt.ylabel("Erro (MSE)")
plt.grid(True)
plt.show()
