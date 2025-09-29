# Módulo: preparacao_dados.py
import pandas as pd
import numpy as np
from typing import Tuple

def preencher_valores_ausentes(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Preenche dados faltantes usando mediana para colunas numéricas
    e moda para colunas categóricas.
    """
    df_processado = dataframe.copy()
    for coluna in df_processado.columns:
        if pd.api.types.is_numeric_dtype(df_processado[coluna]):
            if df_processado[coluna].isnull().any():
                mediana = df_processado[coluna].median()
                df_processado[coluna].fillna(mediana, inplace=True)
        else:
            if df_processado[coluna].isnull().any():
                moda = df_processado[coluna].mode()[0]
                df_processado[coluna].fillna(moda, inplace=True)
    return df_processado

def dividir_treino_teste_estratificado(X: pd.DataFrame, y: pd.Series, percentual_teste=0.2, seed_aleatoria=42) -> Tuple:
    """
    Divide os dados em treino e teste de forma estratificada para manter
    a proporção das classes.
    """
    indices_treino, indices_teste = train_test_split_stratified(y, test_size=percentual_teste, seed=seed_aleatoria)
    X_treino, X_teste = X.iloc[indices_treino], X.iloc[indices_teste]
    y_treino, y_teste = y.iloc[indices_treino], y.iloc[indices_teste]
    return X_treino, X_teste, y_treino, y_teste

def discretizar_por_quantis(coluna: pd.Series, num_faixas: int = 4) -> pd.Series:
    """
    Converte uma coluna contínua em categórica, dividindo-a em faixas
    com o mesmo número de amostras (quantis).
    """
    faixas = pd.qcut(coluna, q=num_faixas, labels=False, duplicates='drop')
    return faixas.astype(str)