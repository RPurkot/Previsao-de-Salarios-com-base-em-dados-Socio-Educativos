# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:18:42 2025

@author: ra058832
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import boxcox
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Carregar os dados
df = pd.read_csv("Large_Simulated_Salary_Data.csv")

# Exibir informações iniciais
def explorar_dados(df):
    print("\n### Informações do Dataset ###")
    print(df.info())
    print("\n### Estatísticas Descritivas ###")
    print(df.describe())
    print("\n### Primeiras Linhas ###")
    print(df.head())

explorar_dados(df)

# Pré-processamento: Codificação One-Hot e Normalização
def preprocessar_dados(df):
    categorical_features = ["Formação", "Sexo", "Raça"]
    encoder = OneHotEncoder(drop="first", sparse=False)
    encoded_cats = encoder.fit_transform(df[categorical_features])
    encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_features))
    
    df_processed = pd.concat([df.drop(columns=categorical_features), encoded_df], axis=1)
    
    scaler = StandardScaler()
    df_processed[["Idade"]] = scaler.fit_transform(df_processed[["Idade"]])
    
    return df_processed

df_processed = preprocessar_dados(df)

# Definição das variáveis independentes e dependente
X = df_processed.drop(columns=["Salário"])
y = df_processed["Salário"]

# Adicionar uma constante ao modelo
X_ols = sm.add_constant(X)

# Ajustar a regressão OLS
ols_model = sm.OLS(y, X_ols).fit()
print("\n### Resumo do Modelo OLS (Sem Transformação) ###")
print(ols_model.summary())

# Avaliação da normalidade dos resíduos
residuos = ols_model.resid
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(residuos, bins=50, kde=True, ax=axes[0])
axes[0].set_title("Distribuição dos Resíduos")
sm.qqplot(residuos, line='s', ax=axes[1])
axes[1].set_title("QQ-Plot dos Resíduos")
plt.tight_layout()
plt.show()

# Aplicação da Transformação Box-Cox
y_positive = y + 1
y_boxcox, lambda_bc = boxcox(y_positive)

X_ols_boxcox = sm.add_constant(X)
ols_model_boxcox = sm.OLS(y_boxcox, X_ols_boxcox).fit()
print("\n### Resumo do Modelo OLS (Com Transformação Box-Cox) ###")
print(ols_model_boxcox.summary())
print(f"\nLambda Box-Cox encontrado: {lambda_bc}")

# Avaliação dos resíduos após Box-Cox
residuos_boxcox = ols_model_boxcox.resid
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(residuos_boxcox, bins=50, kde=True, ax=axes[0])
axes[0].set_title("Distribuição dos Resíduos Após Box-Cox")
sm.qqplot(residuos_boxcox, line='s', ax=axes[1])
axes[1].set_title("QQ-Plot dos Resíduos Após Box-Cox")
plt.tight_layout()
plt.show()

# Função para prever salário

def prever_salario(idade, formacao, sexo, raca):
    novo_dado = pd.DataFrame({
        "Idade": [(idade - df["Idade"].mean()) / df["Idade"].std()],
        "Formação_Ensino Fundamental": [1 if formacao == "Ensino Fundamental" else 0],
        "Formação_Ensino Médio": [1 if formacao == "Ensino Médio" else 0],
        "Formação_Graduação": [1 if formacao == "Graduação" else 0],
        "Formação_Mestrado": [1 if formacao == "Mestrado" else 0],
        "Formação_Pós-Graduação": [1 if formacao == "Pós-Graduação" else 0],
        "Sexo_Masculino": [1 if sexo == "Masculino" else 0],
        "Raça_Branco": [1 if raca == "Branco" else 0],
        "Raça_Indígena": [1 if raca == "Indígena" else 0],
        "Raça_Negro": [1 if raca == "Negro" else 0],
        "Raça_Pardo": [1 if raca == "Pardo" else 0]
    })
    
    novo_dado.insert(0, 'const', 1)
    salario_predito_original = ols_model.predict(novo_dado)[0]
    salario_predito_boxcox = ols_model_boxcox.predict(novo_dado)[0]
    salario_predito_boxcox_original = (salario_predito_boxcox * lambda_bc + 1) ** (1 / lambda_bc)
    
    return salario_predito_original, salario_predito_boxcox_original

# Exemplo de Previsões
print("\nExemplo de Previsões:")
salario1_original, salario1_boxcox = prever_salario(27, "Pós-Graduação", "Masculino", "Branco")
print(f"Homem branco, 27 anos, pós-graduação:\n  - OLS Original: R$ {salario1_original:.2f}\n  - OLS Box-Cox: R$ {salario1_boxcox:.2f}")

salario2_original, salario2_boxcox = prever_salario(18, "Ensino Fundamental", "Feminino", "Indígena")
print(f"Mulher indígena, 18 anos, ensino fundamental:\n  - OLS Original: R$ {salario2_original:.2f}\n  - OLS Box-Cox: R$ {salario2_boxcox:.2f}")
