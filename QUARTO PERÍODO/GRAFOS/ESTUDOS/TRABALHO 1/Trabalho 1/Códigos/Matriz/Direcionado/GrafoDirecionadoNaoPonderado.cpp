#include <iostream>
#include <chrono>
#include <queue>
#include <vector>
using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

void IniciarGrafo_DNP(bool **grafo, int vert){
    for(int i=0;i<vert;i++){
        grafo[i] = new bool[vert];
        for(int j=0;j<vert;j++){
            grafo[i][j] = false;
        }
    }
}

void CriarArestas_DNP(bool **grafo, int vert){
    cout << "Digite os dois vertices que vao ser conectados (Digite 0 0 para sair)." << endl;
    int v1,v2;
    int maxvert = (vert*(vert-1))/2;
    while (maxvert > 0)
    {
        cin >> v1 >> v2;

        if (v1 == 0 && v2 == 0)
        {
            break;
        }

        if (v1 == v2)
        {
            cout << "Loops nao existem em um grafo simples! Por favor escreva uma aresta valida." << endl;
            continue;
        }

        if (v1 < 0 || v1 >= vert || v2 < 0 || v2 >= vert)
        {
            cout << "Vertices invalidas! Por favor escreva arestas entre A e " << vert - 1 << endl;
            continue;
        }

        grafo[v1][v2] = true;
        maxvert -= 1;
    }
}

void ImprimirGrafo_DNP(bool **grafo, int vert){
    char letra = 'A';   
    cout << "  ";
    for (int i = 0; i < vert; i++) {
        cout << letra << " ";
        letra += 1;     
    }
    cout << endl;
    letra = 'A';
    for(int i=0;i<vert;i++){
        cout << letra << " ";
        for(int j=0;j<vert;j++){
            cout << grafo[i][j] << " ";
        }
        cout << endl;
        letra += 1;
    }
}

void DFS_DNP(bool **grafo, int vert, int atual, bool *visitado) {
    visitado[atual] = true;
    for (int i = 0; i < vert; i++) {
        if (grafo[atual][i] && !visitado[i]) {
            DFS_DNP(grafo, vert, i, visitado);
        }
    }
}

void BFS_DNP(bool **grafo, int vert, int origem, int destino) {
    vector<int> distancia(vert, -1);
    vector<int> anterior(vert, -1);
    queue<int> fila;

    distancia[origem] = 0;
    fila.push(origem);

    while (!fila.empty()) {
        int atual = fila.front();
        fila.pop();

        for (int i = 0; i < vert; i++) {
            if (grafo[atual][i] && distancia[i] == -1) {
                distancia[i] = distancia[atual] + 1;
                anterior[i] = atual;
                fila.push(i);
                if (i == destino) break;
            }
        }
    }

    if (distancia[destino] == -1) {
        cout << "Nao existe caminho entre " << origem << " e " << destino << endl;
    } else {
        cout << "Menor caminho tem " << distancia[destino] << " arestas." << endl;
        // Imprime o caminho
        vector<int> caminho;
        for (int v = destino; v != -1; v = anterior[v])
            caminho.push_back(v);
        cout << "Caminho: ";
        for (int i = caminho.size() - 1; i >= 0; i--)
            cout << caminho[i] << (i ? " -> " : "\n");
    }
}

int main() {
    int qntVertices;
    cin >> qntVertices;

    bool **grafo = new bool*[qntVertices];
    IniciarGrafo_DNP(grafo, qntVertices);

    CriarArestas_DNP(grafo, qntVertices);

    int origem, destino;
    cout << "Digite o vertice de origem e destino para o BFS: ";
    cin >> origem >> destino;

    auto inicio = high_resolution_clock::now();
    BFS_DNP(grafo, qntVertices, origem, destino);
    auto fim = high_resolution_clock::now();
    auto duracao_ms = duration_cast<milliseconds>(fim - inicio);
    cout << "Tempo: " << duracao_ms.count() << " ms" << endl;

    // Libera memória
    for (int i = 0; i < qntVertices; i++)
        delete[] grafo[i];
    delete[] grafo;
}