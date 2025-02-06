# Previsao-de-Salarios-com-base-em-dados-Socio-Educativos
NOTA: os dados desta simulação são fictícios (gerados por IA) e possuem apenas caráter didático.

# Análise de Regressão OLS com Transformação Box-Cox

## 📌 Descrição do Projeto
Este repositório contém uma análise completa de regressão linear (OLS) para prever salários com base em características individuais, incluindo idade, formação, sexo e raça. Além disso, utilizamos a **transformação Box-Cox** para melhorar a normalidade dos resíduos e comparar os modelos antes e depois da transformação.

## 📊 Técnicas Utilizadas
- **Exploração de Dados**: Estatísticas descritivas e análise inicial do dataset.
- **Pré-processamento**:
  - Codificação One-Hot para variáveis categóricas.
  - Normalização da idade.
- **Regressão Linear (OLS)**:
  - Ajuste do modelo sem transformação.
  - Avaliação da normalidade dos resíduos (histograma e QQ-Plot).
  - Aplicação da transformação **Box-Cox** para corrigir a normalidade.
  - Comparação dos modelos antes e depois da transformação.
- **Previsões**:
  - Implementação de uma função para prever o salário com base em diferentes perfis.

## 🛠️ Requisitos
Para executar este projeto, você precisará das seguintes bibliotecas Python:

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn scipy
```

## 🚀 Como Rodar o Código
1. **Baixe os arquivos** e certifique-se de que o dataset `Large_Simulated_Salary_Data.csv` está na mesma pasta do script.
2. **Execute o script Python**:

```bash
python analise_regressao_ols.py
```

3. O script irá:
   - Exibir estatísticas descritivas.
   - Treinar o modelo de regressão linear.
   - Avaliar a normalidade dos resíduos.
   - Aplicar a transformação Box-Cox.
   - Comparar os resultados antes e depois da transformação.
   - Fazer previsões de salário para perfis específicos.

## 📈 Exemplos de Previsões
O código prevê salários para diferentes perfis. Exemplo de saída:

```
Homem branco, 27 anos, pós-graduação:
  - OLS Original: R$ 8,288.18
  - OLS Box-Cox: R$ 8,421.30

Mulher indígena, 18 anos, ensino fundamental:
  - OLS Original: R$ 1,421.20
  - OLS Box-Cox: R$ 1,500.45
```

## 📌 Conclusão
Este projeto demonstra como a regressão linear pode ser usada para prever salários, destacando a importância de normalizar os resíduos para garantir resultados estatisticamente válidos. Além disso, evidencia impactos sociais nos salários, como diferença de renda por gênero e raça.

Sinta-se à vontade para contribuir, relatar problemas ou sugerir melhorias! 🚀

