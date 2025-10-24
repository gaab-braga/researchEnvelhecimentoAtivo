# Índice Brasileiro de Envelhecimento Ativo (AAI-BR) — Análise Completa (PNS 2019)

> Análise abrangente do envelhecimento ativo no Brasil, utilizando dados da Pesquisa Nacional de Saúde (PNS 2019). Inclui ETL, construção de indicadores, análises estatísticas avançadas, modelagem preditiva e recomendações para políticas públicas.

## Tabela de Conteúdos

- [Visão Geral e Objetivos](#visão-geral-e-objetivos)
- [Arquitetura do Projeto](#arquitetura-do-projeto)
- [Metodologia e Pipeline](#metodologia-e-pipeline)
- [Análises Realizadas](#análises-realizadas)
- [Dicionário de Dados e Métricas](#dicionário-de-dados-e-métricas)
- [Execução e Ambiente](#execução-e-ambiente)
- [Entregáveis e Validação](#entregáveis-e-validação)
- [Limitações e Próximos Passos](#limitações-e-próximos-passos)
- [Contato e Citação](#contato-e-citação)

## Visão Geral e Objetivos

Este projeto desenvolve um **Índice de Envelhecimento Ativo (AAI)** para o Brasil, baseado nos dados da Pesquisa Nacional de Saúde (PNS 2019). O objetivo é medir a qualidade de vida dos idosos brasileiros através de quatro pilares principais:

1. **Saúde**: Capacidade funcional e controle de doenças crônicas
2. **Participação Social**: Engajamento comunitário e acesso a redes de apoio
3. **Segurança Econômica**: Renda adequada e proteção social
4. **Acesso a Serviços**: Uso de serviços de saúde e conectividade digital

### Impacto Social

O Brasil tem mais de 30 milhões de pessoas com 60 anos ou mais. Este estudo identifica:
- Municípios prioritários para intervenção
- Grupos populacionais mais vulneráveis
- Fatores que mais influenciam o envelhecimento saudável
- Recomendações práticas para gestores públicos

### Abordagem Científica

- **Dados survey-aware**: Considera pesos amostrais e desenho complexo da PNS
- **Intervalos de confiança**: Bootstrap para estimativas robustas
- **Análises avançadas**: Clustering, modelagem preditiva, análise espacial
- **Reprodutibilidade**: Código versionado e documentado

## Arquitetura do Projeto

```
researchEnvelhecimentoAtivo/
├── data/
│   ├── raw/
│   │   ├── PNS_2019.txt                     # Microdados originais (IBGE)
│   │   ├── input_PNS_2019.sas               # Layout das colunas
│   │   └── metadados_core.txt               # Descrição do dataset
│   └── processed/
│       ├── pns_2019_pandas.csv              # Dataset processado (Pandas)
│       ├── pns_2019_spark.csv               # Dataset processado (Spark)
│       ├── BR_Municipios_2019.*             # Shapefiles para análise espacial
│       └── pns_mappings.json                # Mapeamentos categóricos
├── scripts/
│   ├── pns_2019_pandas.py                   # ETL em Pandas
│   └── pns_2019_spark.py                    # ETL em PySpark
├── outputs_aai/
│   ├── municipal_scores_with_ci.csv         # Scores por município
│   ├── priority_municipalities_bottom20.csv # Municípios prioritários
│   ├── aging_profiles.csv                   # Perfis de envelhecimento
│   ├── feature_importance.csv               # Importância das variáveis
│   ├── policy_brief_automated.txt           # Policy brief automatizado
│   ├── shap_values.csv                      # Valores SHAP
│   └── *.png                                # Visualizações
├── EDA.ipynb                                # Notebook principal de análise
├── colunas_faltantes.md                     # Documentação de colunas
└── README.md                                # Este arquivo
```

## Metodologia e Pipeline

### 1. Extração e Limpeza (ETL)

- **Fonte**: Microdados PNS 2019 (IBGE)
- **Filtro**: Indivíduos com 60 anos ou mais
- **Limpeza**: Tratamento de valores missing, codificações e coerções
- **Validação**: Verificação de pesos amostrais e consistência

### 2. Construção dos Indicadores

#### Domínios do AAI
- **Health Score**: Combinação de autoavaliação de saúde, multimorbidade e funcionalidade
- **Functional Score**: Capacidade para atividades diárias (ADL/IADL)
- **Participation Score**: Acesso à internet e celular
- **Economic Score**: Educação e renda
- **Access Score**: Plano de saúde e consultas médicas

#### AAI Total
- **Média ponderada** dos domínios normalizados (0-1)
- **Pesos iguais** (baseline) ou definidos por especialistas

### 3. Análises Estatísticas

- **Agregação municipal**: Scores por cidade com controle de qualidade
- **Análise de desigualdades**: Por sexo, raça, escolaridade, urbano/rural
- **Clustering**: Identificação de perfis de envelhecimento
- **Modelagem preditiva**: Fatores de vulnerabilidade (Random Forest + SHAP)
- **Análise de mediação**: Efeitos indiretos entre variáveis
- **Análise espacial**: Padrões geográficos e autocorrelação

## Análises Realizadas

O notebook `EDA.ipynb` contém análises completas organizadas em seções:

1. **Preparação dos Dados**: Limpeza e validação
2. **Construção do AAI**: Cálculo dos indicadores
3. **Agregação Municipal**: Scores por município
4. **Identificação de Hotspots**: Municípios prioritários (20% piores)
5. **Análise de Desigualdades**: Diferenças entre subgrupos
6. **Perfis de Envelhecimento**: Clustering em 4 grupos
7. **Modelagem Preditiva**: Drivers de vulnerabilidade
8. **Análise de Mediação**: Efeitos indiretos
9. **Análise Espacial**: Padrões geográficos
10. **Policy Brief**: Recomendações automatizadas
11. **Visualizações**: Gráficos e mapas
12. **Validação**: Checklist de qualidade
13. **Conclusões**: Sumário executivo
14. **Export**: Datasets finais

### Principais Resultados

- **AAI Nacional**: Média brasileira com intervalos de confiança
- **Hotspots**: Lista de municípios prioritários
- **Perfis**: 4 grupos distintos de idosos
- **Fatores Críticos**: Idade, escolaridade, uso de medicamentos
- **Padrões Espaciais**: Autocorrelação significativa
- **Recomendações**: Prioridades para curto, médio e longo prazo

## 📊 Dicionário de Dados e Métricas

### 🔍 Colunas do DataFrame Principal (`df`)

| Nome Técnico | O que Significa (Explicação Simples) |
|--------------|-------------------------------------|
| `peso` / `peso_amostral` | **Fator de Multiplicação**: Número que indica quantas pessoas na população brasileira aquele entrevistado representa. Essencial para validade nacional. |
| `codmun` | **Código do Município**: Código oficial do IBGE para a cidade de residência. |
| `uf` | **Unidade Federativa**: Sigla do estado (ex: SP, RJ, BA). |
| `estrato` / `upa` | **Informação de Amostragem**: Detalhes técnicos de seleção. Garante representatividade estatística. |
| `faixa_etaria` | **Grupo de Idade**: Idade categorizada em grupos (ex: '60-69', '70-79'). |
| `anos_estudo` | **Anos de Escolaridade**: Tempo de estudo formal. |
| `renda` / `renda_percapita` | **Renda**: Rendimento financeiro individual ou familiar. |
| `health_score` | **Nota de Saúde**: Índice 0-100 combinando doenças crônicas e saúde geral. |
| `functional_score` | **Nota de Capacidade Funcional**: Índice 0-100 medindo atividades diárias (caminhar, subir escadas). |
| `participation_score` | **Nota de Participação e Conexão**: Mede acesso a internet e celular. |
| `econ_score` | **Nota de Segurança Econômica**: Combina renda e escolaridade. |
| `access_score` | **Nota de Acesso à Saúde**: Mede plano de saúde e consultas recentes. |
| **`AAI_total`** | **Índice de Envelhecimento Ativo (Nota Final)**: Média de todas as notas acima. Principal indicador do estudo. |
| `cluster` | **Perfil de Envelhecimento**: Grupo identificado por características (ex: 'Ativos e Conectados'). |
| `vulnerable` | **Indicador de Vulnerabilidade**: Etiqueta identificando 20% com piores notas. |

### 🏛️ Métricas da Análise Municipal (`municipal_scores`)

| Nome Técnico | O que Significa (Explicação Simples) |
|--------------|-------------------------------------|
| `n_obs` | **Número de Entrevistados**: Quantidade de pessoas entrevistadas na cidade. Mede confiabilidade da nota. |
| `pop_weight_sum` | **População Estimada**: Estimativa de idosos na cidade usando fatores de multiplicação. |
| `AAI_ci_lower` | **Piso do Intervalo de Confiança**: Valor mínimo provável para a média da cidade. |
| `AAI_ci_upper` | **Teto do Intervalo de Confiança**: Valor máximo provável para a média da cidade. |
| `reliable` | **Selo de Confiabilidade**: Etiqueta indicando se a nota é confiável baseado em `n_obs`. |

### 🔬 Termos Técnicos das Análises

| Nome Técnico | O que Significa (Explicação Simples) |
|--------------|-------------------------------------|
| `Bootstrap` | **Teste de Estabilidade**: Técnica repetindo análise centenas de vezes com amostras aleatórias para garantir resultados não são acaso. |
| `Moran's I` | **Índice de Vizinhança**: Métrica mostrando se cidades vizinhas têm notas parecidas (formando "manchas" no mapa). |
| `p-valor` | **Teste de Sorte**: Probabilidade do padrão encontrado ser acaso. Valor baixo (< 0.05) significa padrão provavelmente real. |
| `LISA` / `Cluster Espacial` | **Identificador de Hotspots**: Análise identificando "panelinhas" de municípios (ex: `LL` = cidades com notas baixas cercadas por vizinhos também baixos). |
| `Feature Importance` | **Ranking de Influência**: Lista mostrando fatores mais importantes para prever vulnerabilidade (ex: renda, escolaridade). |
| `SHAP Values` | **Explicador de Influência**: Técnica mostrando não só *quais* fatores importam, mas *como* influenciam cada pessoa. |
| `Análise de Mediação` | **Análise de Efeito Indireto**: Investiga se fator A afeta C diretamente ou através de intermediário B. |

## Execução e Ambiente

### Dependências

```bash
# Python 3.8+
pip install pandas numpy scikit-learn statsmodels seaborn matplotlib plotly
pip install geopandas libpysal esda  # Para análise espacial
pip install shap  # Para interpretabilidade
pip install pyspark  # Para versão Spark (opcional)
```

### Como Executar

1. **Clone o repositório** e instale dependências
2. **Execute o ETL** (se necessário):
   ```bash
   python scripts/pns_2019_pandas.py
   ```
3. **Abra o notebook principal**:
   ```bash
   jupyter notebook EDA.ipynb
   ```
4. **Execute as células** sequencialmente (leva ~30-60 minutos)
5. **Verifique os outputs** na pasta `outputs_aai/`

### Ambiente Recomendado

- **Python**: 3.8 ou superior
- **Memória**: 8GB+ recomendado
- **Espaço**: ~5GB para dados e outputs
- **Sistema**: Windows/Linux/Mac

## Entregáveis e Validação

### Outputs Principais

- **Dataset Processado**: `pns_2019_processed_60plus.csv`
- **Scores Municipais**: `municipal_scores_with_ci.csv`
- **Municípios Prioritários**: `priority_municipalities_bottom20.csv`
- **Perfis de Envelhecimento**: `aging_profiles.csv`
- **Policy Brief**: `policy_brief_automated.txt`
- **Visualizações**: Arquivos PNG com gráficos
- **Análise Espacial**: `municipal_aai_spatial.geojson`

### Checklist de Validação

O notebook inclui validações automáticas:
- ✅ Filtro de idade aplicado corretamente
- ✅ Pesos amostrais utilizados
- ✅ AAI calculado com domínios disponíveis
- ✅ Agregação municipal com intervalos de confiança
- ✅ Hotspots filtrados por confiabilidade
- ✅ Modelos treinados e avaliados
- ✅ Outputs salvos corretamente

## Limitações e Próximos Passos

### Limitações

- **Dados transversais**: Não permitem inferir causalidade
- **Auto-relato**: Alguns indicadores sujeitos a viés de memória
- **Cobertura municipal**: Cidades com poucos entrevistados têm estimativas menos precisas
- **Temporal**: Dados de 2019; atualizações futuras necessárias

### Próximos Passos

- **Dashboard Interativo**: Interface web para exploração dos dados
- **Análises Longitudinais**: Comparação com PNS 2013 e futuras
- **Integração com Outros Dados**: Merge com DATASUS, Censo, etc.
- **Modelo Preditivo Aprimorado**: Deep Learning para identificação de risco
- **Validação com Especialistas**: Painel Delphi para pesos do índice

## Contato e Citação

**Autor**: Gabriel Braga (gaab-braga)
**Repositório**: https://github.com/gaab-braga/researchEnvelhecimentoAtivo
**Data**: Outubro 2025
**Licença**: MIT

**Como citar**:
```
Braga, G. (2025). Índice Brasileiro de Envelhecimento Ativo (AAI-BR): Análise da PNS 2019.
GitHub repository: https://github.com/gaab-braga/researchEnvelhecimentoAtivo
```

---

*Este projeto contribui para o debate sobre políticas públicas para o envelhecimento populacional no Brasil, fornecendo evidências científicas para decisões informadas.*