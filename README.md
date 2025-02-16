# Análise de Sentimento e Modelagem Econométrica: Uma Abordagem via Twitter

Este repositório contém o código-fonte e materiais relacionados à minha tese de doutorado, que explora a relação entre sentimentos expressos nas redes sociais (twitter) e variáveis econômicas através de técnicas avançadas de mineração de dados e econometria.

## 📋 Visão Geral

O projeto está estruturado em três componentes principais:

1. **Extração e Mineração de Dados do Twitter**
   - Coleta de dados via API do Twitter
   - Pré-processamento e limpeza dos tweets
   - Análise inicial dos dados coletados

2. **Processamento de Linguagem Natural e Índice de Sentimento**
   - Implementação de técnicas de NLP para análise de sentimento
   - Construção de índices de sentimento
   - Validação e calibração dos índices

3. **Modelagem Econométrica**
   - Implementação de modelos VAR (Vector Autoregression)
   - Análise via Local Projections
   - Avaliação do impacto do sentimento em variáveis econômicas por meio de Funções de Impulso Resposta (FRI)

## 🚀 Estrutura do Projeto

```
├── Exports/              # Dataframes gerados pelo código
├── Language/             # Recursos linguísticos utilizados como input pelos algoritmos
├── Statistic/            # Métodos estatísticos sem lib correspondente (i.e. Método de dessazonalização X12AS)
└── Storage/              # Dados minerados no twitter
└── Economic/             # Dados econômicos utilizados como input (muito pesado para o github)
└── Notebooks/            # Notebooks com instruções de como utilizar os dados e o código
```

## 🛠️ Tecnologias Utilizadas

- Python para processamento de dados e NLP
- R para análises econométricas


## 📊 Metodologia

O projeto segue um pipeline de dados que integra:
1. Mineração de dados do Twitter
2. Processamento e análise de sentimento via NLP
3. Construção de índices de sentimento
4. Aplicação em modelos econométricos

## 🔍 Em Desenvolvimento

Este repositório está evolução, com atualizações ainda a serem incluídas nos seguintes aspectos:
- Instruções para replicação do código: Atualizar o README.md após mudanças abaixo
- Centralizar todos os inputs em apenas um diretório (Language, Statistic, Storage, leia.py)
- Orquestrar um pipeline automático de dados (extrair, preprocessing, sentiment analysis)
- Finalizar centralização de todas as variáveis de ambiente (statistical_analysis ainda tem paths próprios)
- Inclusão de um diretório de notebooks com exemplos de como operar as principais funções
- criar requirements.txt ao final do processo com dependências necessárias


## 📫 Contato

rsnatorres@gmail.com

