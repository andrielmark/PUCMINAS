#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#define MAX 100



void gerarmatriz(int matrizgrafo[max][max], int vertices){
    for(int i = 0; i < vertices; i++){
        for(int j = 0; j < vertices; j++){
            matrizgrafo[i][j] = 0;
        }
    }
}

int main(){
    int vertices = 0; int verticeorigem = 0; int verticedestino = 0; int qntaresta = 0;
    scanf("%d", &vertices);
    int matrizgrafo[max][max];
    gerarmatriz(matrizgrafo, vertices);
    scanf("%d", &qntaresta);
    while(qntaresta > 0){
        scanf("%d", &verticeorigem);
        scanf("%d", &verticedestino);
        matrizgrafo[verticeorigem][verticedestino] = 1;
        matrizgrafo[verticedestino][verticeorigem] = 1;
        qntaresta--;
    }
    return 0;
}
