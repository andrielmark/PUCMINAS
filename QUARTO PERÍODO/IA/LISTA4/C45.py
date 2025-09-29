# -*- coding: utf-8 -*-
"""
Implementação do Algoritmo C4.5
Autor: Adnriel Mark da Silva Pinto
Matrícula: 1528633

- Critério de Divisão: Razão de Ganho (Gain Ratio)
- Atributos Contínuos: Suportados nativamente através da busca por limiar.
- Estrutura da Árvore: Multi-ramificada (nós binários para contínuos).
"""

import numpy as np
import pandas as pd
from collections import Counter

# ==================================
# Estrutura do Nó da Árvore C4.5
# ==================================
class C45Node:
    def __init__(self, feature_name=None, threshold=None, branches=None, prediction=None):
        self.feature_name = feature_name
        self.threshold = threshold  # Apenas para atributos contínuos
        self.branches = branches or {}  # Dicionário de filhos
        self.prediction = prediction # Rótulo, se for um nó folha

    def is_leaf(self):
        return self.prediction is not None

    def __str__(self, level=0):
        indent = "  |   " * level
        if self.is_leaf():
            return f"{indent}Folha -> Predição: {self.prediction}\n"
        
        condition = f"Dividindo por '{self.feature_name}' com limiar < {self.threshold:.4f}"
        s = f"{indent}{condition}\n"
        for branch_name, child_node in self.branches.items():
            s += f"{indent}  --> Ramo '{branch_name}':\n{child_node.__str__(level + 1)}"
        return s

# ==================================
# Funções do Algoritmo C4.5
# ==================================
def calculate_entropy(target_values):
    """Calcula a entropia de um array de rótulos."""
    _, counts = np.unique(target_values, return_counts=True)
    probabilities = counts / len(target_values)
    return -np.sum(probabilities * np.log2(probabilities))

def calculate_gain_ratio(feature_values, target_values, threshold):
    """Calcula a Razão de Ganho para uma divisão binária."""
    parent_entropy = calculate_entropy(target_values)
    
    # Dividir os dados
    left_indices = feature_values < threshold
    right_indices = feature_values >= threshold
    
    y_left, y_right = target_values[left_indices], target_values[right_indices]

    if len(y_left) == 0 or len(y_right) == 0:
        return 0

    # Calcular entropia ponderada dos filhos
    p_left = len(y_left) / len(target_values)
    p_right = len(y_right) / len(target_values)
    child_entropy = p_left * calculate_entropy(y_left) + p_right * calculate_entropy(y_right)
    
    # Ganho de Informação
    information_gain = parent_entropy - child_entropy
    
    # Split Info
    split_info = - (p_left * np.log2(p_left) + p_right * np.log2(p_right))
    if split_info == 0:
        return 0
        
    return information_gain / split_info

def find_best_split_c45(X_data, y_data, feature_names):
    """Encontra o melhor atributo e limiar para dividir os dados."""
    best_gain_ratio = -1
    best_feature_index = None
    best_threshold = None

    n_features = X_data.shape[1]
    
    for feat_idx in range(n_features):
        feature_col = X_data[:, feat_idx]
        unique_vals = np.unique(feature_col)
        
        if len(unique_vals) > 1:
            # Testa pontos médios como candidatos a limiar
            threshold_candidates = (unique_vals[:-1] + unique_vals[1:]) / 2
            for threshold in threshold_candidates:
                gain_ratio = calculate_gain_ratio(feature_col, y_data, threshold)
                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    best_feature_index = feat_idx
                    best_threshold = threshold

    if best_gain_ratio == -1:
        return None, None, None
        
    return best_feature_index, best_threshold, feature_names[best_feature_index]

def build_c45_tree(X_data, y_data, feature_names, max_depth=None, current_depth=0):
    """Constrói a árvore C4.5 de forma recursiva."""
    # Condições de parada
    if len(np.unique(y_data)) == 1:
        return C45Node(prediction=y_data[0])

    if X_data.shape[1] == 0 or (max_depth is not None and current_depth >= max_depth):
        most_common = Counter(y_data).most_common(1)[0][0]
        return C45Node(prediction=most_common)

    best_feat_idx, best_thresh, best_feat_name = find_best_split_c45(X_data, y_data, feature_names)

    if best_feat_idx is None:
        most_common = Counter(y_data).most_common(1)[0][0]
        return C45Node(prediction=most_common)

    # Dividir dados e construir sub-árvores
    branches = {}
    left_mask = X_data[:, best_feat_idx] < best_thresh
    right_mask = ~left_mask
    
    branches['<'] = build_c45_tree(X_data[left_mask, :], y_data[left_mask], feature_names, max_depth, current_depth + 1)
    branches['>='] = build_c45_tree(X_data[right_mask, :], y_data[right_mask], feature_names, max_depth, current_depth + 1)

    return C45Node(feature_name=best_feat_name, threshold=best_thresh, branches=branches)

def classify_sample(sample, tree_node: C45Node):
    """Classifica uma nova amostra usando a árvore treinada."""
    if tree_node.is_leaf():
        return tree_node.prediction
    
    # Supondo que a amostra é um dicionário {nome_feature: valor}
    value = sample[tree_node.feature_name]
    
    branch_key = '<' if value < tree_node.threshold else '>='
    if branch_key in tree_node.branches:
        return classify_sample(sample, tree_node.branches[branch_key])
    else:
        # Caso um ramo não exista (poda ou dados de teste diferentes)
        # Retorna a predição majoritária do nó atual (simplificação)
        return Counter([n.prediction for n in tree_node.branches.values() if n.is_leaf()]).most_common(1)[0][0]

if __name__ == "__main__":
    print("Executando o Algoritmo C4.5 de Adnriel Pinto...")
    # Exemplo de como usar:
    # df = pd.read_csv('seu_dataset.csv')
    # feature_names = list(df.columns[:-1])
    # X_data = df.iloc[:, :-1].values
    # y_data = df.iloc[:, -1].values
    # tree = build_c45_tree(X_data, y_data, feature_names, max_depth=5)
    # print(tree)