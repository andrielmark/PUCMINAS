# ============================================================
# Questão 2 – Reconhecimento de Dígitos (Display de 7 Segmentos)
# Autor: Andriel Mark da Silva Pinto — Matrícula: 1528633
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# Dados de entrada (7 segmentos) e saída one-hot (4 bits)
X = np.array([
    [1,1,1,1,1,1,0],  # 0
    [0,1,1,0,0,0,0],  # 1
    [1,1,0,1,1,0,1],  # 2
    [1,1,1,1,0,0,1],  # 3
    [0,1,1,0,0,1,1],  # 4
    [1,0,1,1,0,1,1],  # 5
    [1,0,1,1,1,1,1],  # 6
    [1,1,1,0,0,0,0],  # 7
    [1,1,1,1,1,1,1],  # 8
    [1,1,1,1,0,1,1]   # 9
])

y = np.array([
    [1,0,0,0],  # classe 0
    [0,1,0,0],  # classe 1
    [0,0,1,0],  # classe 2
    [0,0,0,1],  # classe 3
    [1,0,0,0],  # classe 4
    [0,1,0,0],  # classe 5
    [0,0,1,0],  # classe 6
    [0,0,0,1],  # classe 7
    [1,0,0,0],  # classe 8
    [0,1,0,0]   # classe 9
])

# Estrutura da rede
INPUT_LAYER_SIZE = 7
HIDDEN_LAYER_SIZE = 5
OUTPUT_LAYER_SIZE = 4
EPOCHS = 2000
LEARNING_RATE = 0.4

# Funções de ativação
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

# Inicialização dos pesos (alterada para se diferenciar do exemplo anterior)
np.random.seed(10)
W1 = np.random.uniform(-1, 1, (INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE))
B1 = np.zeros((1, HIDDEN_LAYER_SIZE))

W2 = np.random.uniform(-1, 1, (HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE))
B2 = np.zeros((1, OUTPUT_LAYER_SIZE))

errors = []

# ===============================
# Treinamento com Backpropagation
# ===============================
for epoch in range(EPOCHS):
    Z1 = np.dot(X, W1) + B1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + B2
    yHat = sigmoid(Z2)

    error = y - yHat
    loss = np.mean(np.square(error))
    errors.append(loss)

    # Retropropagação
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
# Resultados finais
# ===============================
print("=== RESULTADOS FINAIS (7 SEGMENTOS) ===")
for i in range(len(X)):
    print(f"Entrada {i}: {X[i]} -> Saída prevista: {np.round(yHat[i], 3)}")

# Gráfico do erro
plt.plot(errors)
plt.title("Curva de Erro - Display de 7 Segmentos")
plt.xlabel("Épocas")
plt.ylabel("Erro (MSE)")
plt.grid(True)
plt.show()

# ===============================
# Teste com ruído (simulando falhas)
# ===============================
X_ruido = X + np.random.normal(0, 0.15, X.shape)
Z1r = np.dot(X_ruido, W1) + B1
A1r = sigmoid(Z1r)
Z2r = np.dot(A1r, W2) + B2
yHat_ruido = sigmoid(Z2r)

print("\n=== SAÍDAS COM RUÍDO ===")
for i in range(len(X)):
    print(f"Entrada {i} (com ruído): {np.round(yHat_ruido[i], 3)}")
