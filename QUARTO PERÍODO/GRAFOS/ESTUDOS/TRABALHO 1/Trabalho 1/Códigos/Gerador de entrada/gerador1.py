import random

# --- Parâmetros ---
num_vertices = 5000
num_arestas_total = 10000
nome_arquivo = "grafo_ponderado.txt"

# Usamos um set para registrar as arestas já criadas e evitar duplicatas.
arestas_existentes = set()

print(f"Gerando arquivo '{nome_arquivo}' com um grafo conexo e estrutura randomizada...")
try:
    with open(nome_arquivo, 'w') as f:
        # Escreve a primeira linha
        f.write(f"{num_vertices}\n")

        # --- Fase 1: Garantir Conectividade com Estrutura de Árvore Aleatória ---
        conectados = {0} # Começa com o vértice 0 no conjunto de conectados
        nao_conectados = list(range(1, num_vertices))
        random.shuffle(nao_conectados) # Embaralhar para mais aleatoriedade

        # Lista de vértices já conectados para escolha aleatória eficiente
        lista_conectados = [0]

        while nao_conectados:
            # Pega um vértice aleatório que já faz parte da "árvore"
            v1 = random.choice(lista_conectados)
            
            # Pega o próximo vértice da lista de não conectados
            v2 = nao_conectados.pop()

            peso = random.randint(1, 10)
            
            # Garante que a aresta é canônica (menor, maior)
            aresta_canonica = tuple(sorted((v1, v2)))

            f.write(f"{v1} {v2} {peso}\n")
            arestas_existentes.add(aresta_canonica)
            
            # Adiciona o novo vértice à lista de conectados
            conectados.add(v2)
            lista_conectados.append(v2)

        # --- Fase 2: Adicionar as Arestas Restantes Aleatoriamente ---
        while len(arestas_existentes) < num_arestas_total:
            v1 = random.randint(0, num_vertices - 1)
            v2 = random.randint(0, num_vertices - 1)

            aresta_canonica = tuple(sorted((v1, v2)))
            if v1 != v2 and aresta_canonica not in arestas_existentes:
                peso = random.randint(1, 10)
                f.write(f"{v1} {v2} {peso}\n")
                arestas_existentes.add(aresta_canonica)

    print(f"Arquivo '{nome_arquivo}' gerado com sucesso!")
    print(f"Total de arestas no arquivo: {len(arestas_existentes)}")

except IOError as e:
    print(f"Erro ao escrever o arquivo: {e}")