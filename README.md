# Índice Brasileiro de Envelhecimento Ativo (AAI-BR) — ETL (PNS 2019)

> Pipeline ETL e conjunto de métricas para extrair, limpar e construir indicadores do envelhecimento ativo usando os microdados da PNS-2019, produzindo um dataset reduzido, indicadores por domínio e produtos analíticos adequados para pesquisa e suporte à política pública.

## Tabela de Conteúdos

- [Visão Geral e Objetivos](#visão-geral-e-objetivos)
- [Arquitetura do Projeto](#arquitetura-do-projeto)
- [Metodologia e Pipeline](#metodologia-e-pipeline)
- [Execução e Ambiente](#execução-e-ambiente)
- [Entregáveis e Validação](#entregáveis-e-validação)
- [Limitações e Próximos Passos](#limitações-e-próximos-passos)
- [Contato e Citação](#contato-e-citação)
- [Apêndice: Pseudocódigo](#apêndice-pseudocódigo)

## Visão Geral e Objetivos

Este repositório contém o ETL, documentação e artefatos necessários para transformar os microdados complexos da PNS-2019 em um dataset core (fase-1) e em indicadores compostos que medem o envelhecimento ativo.

O objetivo científico é produzir um índice mensurável e reprodutível, com domínios interpretáveis (Saúde, Funcionalidade, Participação, Capacidade/Educação, Condição Socioeconômica / Acesso), que permita:

- Identificar municípios e grupos populacionais mais vulneráveis (idade, sexo, raça, urbano/rural, escolaridade).
- Decompor o desempenho por domínio para priorização de políticas.
- Prover insumos para análises temporais (quando combinado com PNS 2013) e validação (ELSI, DATASUS, VIGITEL).

## Arquitetura do Projeto

A estrutura de diretórios foi desenhada para garantir a reprodutibilidade e separação entre dados brutos, lógica de transformação e produtos analíticos.

```
pns_2019_etl/
├── data/
│   ├── raw/
│   │   ├── PNS_2019.txt                     # Microdados originais (IBGE)
│   │   ├── input_PNS_2019.sas               # Layout / posições de coluna
│   │   ├── dicionario_PNS_microdados_2019.xls # Dicionário oficial
│   │   └── metadados_core.txt               # Descrição do dataset processado
│   └── processed/
│       ├── pns_2019_core_corrected.csv      # Dataset reduzido final (fase 1)
│       └── pns_mappings.json                # Mapeamentos categóricos
├── scripts/
│   ├── etl_final.py                         # Script ETL principal
│   └── script.py                            # (Versão alternativa PySpark)
├── colunas_escolhidas.md                    # Documentação e justificativa das colunas
└── test.ipynb                               # Notebook para validação e checks
```

<details>
<summary><strong>Descrição dos Componentes Principais</strong></summary>

- **/data/raw**: Contém os arquivos originais e imutáveis fornecidos pelo IBGE. O ETL lê deste diretório.
- **/data/processed**: Contém os outputs do ETL. Estes arquivos são gerados e substituídos programaticamente.
- **/scripts**: Contém toda a lógica de transformação, limpeza e construção de variáveis.
- **colunas_escolhidas.md**: Documento de metadados que justifica a seleção de cada variável com base nos pilares teóricos do envelhecimento ativo.
- **test.ipynb**: Notebook de Análise Exploratória (EDA) e Quality Assurance (QA). Usado para verificar distribuições, missing rates e consistência dos dados processados.

</details>

## Metodologia e Pipeline

### Racional de Seleção de Colunas

As colunas foram selecionadas para traduzir os pilares científicos do envelhecimento ativo (WHO): Saúde, Participação, Capacidade/Educação, Condição Socioeconômica/Segurança e Acesso/Uso de Serviços.

**Critérios de Inclusão (Prioridade):**

- **Relevância Teórica**: Mapeamento direto a um pilar do WHO.
- **Qualidade dos Dados**: Baixa taxa de missing / boa codificação na PNS-2019.
- **Potencial de Derivação**: Capacidade de gerar indicadores robustos (ex.: multimorbidity_count, scores de ADL/IADL).
- **Ação (Policy-making)**: Potencial para agregação (codmun) e suporte a políticas.

<details>
<summary><strong>Lista Consolidada de Variáveis (Core e Derivadas)</strong></summary>

#### Core (Originais PNS)

Nomes conceituais. Mapear para o nome real no microdicionário.

- **Identificação e Desenho Amostral**: `id_ind`, `id_dom`, `codmun`, `uf`, `peso` (amostral), `estrato` (PSU/estrato)
- **Demografia**: `idade`, `sexo`, `raca`, `area_urb`
- **Educação / Renda / Trabalho**: `anos_estudo`, `renda` (per capita), `ocupacao`
- **Moradia / Rede Social**: `mora_sozinho`
- **Saúde Autorreferida e Crônica**: `autoav` (auto-avaliação), `hipert`, `diab`, `cardio`, `avc`, `resp`, `cancer`, `depressao`, `num_meds`
- **Funcionalidade**:
  - ADL: `adl_item_1`, `adl_item_2`, `adl_item_3`
  - IADL: `iadl_item_1`, `iadl_item_2`
  - `queda_12m`
- **Uso e Acesso aos Serviços**: `plano` (possui plano), `usa_sus`, `consulta_12m`, `internacoes_12m`, `vacina_gripe`, `cadastramento_ESF`
- **Estilo de Vida / Prevenção**: `atividade_fisica`, `tabagismo`, `imc`
- **Inclusão Digital**: `uso_internet`, `uso_celular`

#### Derivadas (Construídas no ETL)

| Variável | Fórmula / Descrição |
|----------|---------------------|
| `faixa_etaria_3` | cut(idade, bins=[60–69,70–79,80+]) |
| `multimorbidity_count` | sum(hipert, diab, cardio, avc, resp, cancer, depressao) |
| `multimb_cat` | Categórico de multimorbidity_count (0, 1, 2, 3+) |
| `adl_score` | sum(ADL items) (maior = mais limitação) |
| `iadl_score` | sum(IADL items) |
| `functional_raw` | adl_score + iadl_score |
| `functional_score` | 1 − (functional_raw / max_possible_raw) (Normalizado 0–1; maior = melhor) |
| `dependencia_SUS` | (plano == 0) & (usa_sus == 1) |
| `cobertura_influenza` | vacina_gripe |
| `health_score` | Combinação normalizada (ver Apêndice A) |
| `AAI_dom_*` | Scores normalizados por domínio (Health, Functional, Participation, Economic, Access) |
| `AAI_total` | Agregação dos domínios (método a definir) |

</details>

### Construção de Domínios e Derivados

Abaixo detalhamos a construção dos principais indicadores sintéticos.

<details>
<summary><strong>Saúde e Autonomia Funcional</strong></summary>

**Motivação**: Saúde percebida e carga de doenças crônicas determinam autonomia, uso de serviços e necessidade de cuidados.

**Colunas Core**: `autoav`, `hipert`, `diab`, `cardio`, `avc`, `resp`, `cancer`, `depressao`, `num_meds`.

**Derivados**:

- `multimorbidity_count`: Soma de indicadores binários (0/1).
- `health_score`: Combinação de `autoav`, `multimorbidity_count` e `functional_score`.

**Decisões Técnicas**:

- Antes de combinar, padronizar (z-score) cada componente para neutralizar escalas.
- Inverter sinais quando necessário (ex.: autoav codificado 1=excelente não inverte; multimorbidity_count inverte).
- Normalização final (min-max) para 0-1 para facilitar comparações.

</details>

<details>
<summary><strong>Funcionalidade (ADL / IADL)</strong></summary>

**Motivação**: Limitações nas Atividades de Vida Diária (ADL/IADL) são o núcleo da independência.

**Construção**:

- `adl_score` e `iadl_score` são somas simples (1 = dificuldade).
- `functional_score`: Normaliza e inverte o score bruto. Um `functional_raw` alto (muita dificuldade) leva a um `functional_score` baixo (perto de 0).

**Justificativa**: Scores 0-1 são mais interpretáveis e facilitam agregações por domínio.

</details>

<details>
<summary><strong>Participação Social e Rede de Apoio</strong></summary>

**Motivação**: Participação reduz isolamento e melhora saúde mental.

**Core**: `mora_sozinho`, `participacao_grupos` (se disponível), `frequencia_contatos` (se disponível).

**Derivado**: `indice_participacao` (combinação normalizada de indicadores sociais).

</details>

<details>
<summary><strong>Condição Socioeconômica e Trabalho</strong></summary>

**Motivação**: Renda e educação explicam acesso a serviços e capacidade de resiliência.

**Core**: `anos_estudo`, `renda`, `ocupacao` (participação produtiva).

**Derivados**: `renda_percapita_class` (quintis/quartis) e `participacao_economica` (binário).

</details>

<details>
<summary><strong>Acesso e Dependência do SUS</strong></summary>

**Motivação**: Identificar populações que dependem exclusivamente do sistema público.

**Core**: `plano`, `usa_sus`, `consulta_12m`, `internacoes_12m`, `vacina_gripe`, `cadastramento_ESF`.

**Derivado**: `dependencia_SUS` e `indice_acesso` (combinando frequência de consultas / vacinação).

</details>

<details>
<summary><strong>Regras de Limpeza, Imputação e Coerção</strong></summary>

- **Leitura**: Usar `input_PNS_2019.sas` para parsear o `.txt` (fix-width).
- **Codificações**: Normalizar valores "99/98/97" (não sabe/não respondeu) do dicionário para `NaN`.
- **Coerção**: Converter ordinais (`autoav`) em inteiros; flags Sim/Não em 1/0. Garantir `idade` como inteiro e filtrar amostra (`idade >= 60`).
- **Imputação (Baseline)**: Mediana para numéricos essenciais (`renda`, `anos_estudo`); Moda para categóricos com baixa missingness.
- **Amostra Complexa**: Preservar colunas `peso` e `estrato`. Indicadores agregados (municipais, UF) devem usar estimativas ponderadas.

</details>

### Metodologias de Composição do Índice

O AAI_total será composto pela agregação dos escores de domínio normalizados (AAI_dom_*).

**Pré-processamento**: Dentro de cada domínio, variáveis são transformadas para a mesma direção (maior = melhor) e escala (z-score ou min-max 0–1).

**Estratégias de Ponderação (Agregação)**:

- **Equal weighting (Baseline)**: Média aritmética dos domínios. Simples e transparente para comunicação.
- **PCA / Análise Fatorial (Data-driven)**: Pesos definidos pela variância explicada.
- **Delphi / Expert weights (Recomendado)**: Painel de especialistas define importância relativa.

**Justificativa**: A versão final do índice para política pública deve priorizar a ponderação por especialistas (Delphi) ou pesos iguais (Equal Weighting) devido à interpretabilidade. A análise PCA será usada para validação de sensibilidade.

<details>
<summary><strong>Considerações Metodológicas Avançadas</strong></summary>

- **Governança e Ética**: Os microdados PNS são anonimizados. Nenhuma tabela com células pequenas (risco de reidentificação) deve ser publicada. O código deve ser versionado (Git) e o metadados_core.txt deve registrar o commit SHA da execução do ETL (audit trail).
- **Amostra Complexa**: Para inferência estatística (p-valores, ICs), usar o desenho amostral completo (estratos, psu, pesos) via pacotes de survey (ex: survey em R, statsmodels.survey em Python).
- **Causalidade**: A PNS é transversal. O estudo permite identificar associações ajustadas, não causalidade.

</details>

## Execução e Ambiente

### Dependências

- Python 3.8+
- pandas, numpy, scipy
- scikit-learn (para normalização e PCA)
- statsmodels (para regressões ponderadas)
- pyarrow (para I/O eficiente, ex: parquet)
- pyreadstat (para leitura de layouts SAS, se disponível)

### Instalação

Recomenda-se o uso de um ambiente virtual:

```bash
pip install pandas numpy scikit-learn statsmodels pyarrow
```

### Execução Básica

1. Garanta que os arquivos de `data/raw/` (PNS_2019.txt, input_PNS_2019.sas, dicionario...) estejam presentes.
2. Ajuste os mapeamentos de coluna no `etl_final.py` conforme o dicionário.
3. Execute o script principal:

```bash
python scripts/etl_final.py \
    --input data/raw/PNS_2019.txt \
    --layout data/raw/input_PNS_2019.sas \
    --out data/processed/pns_2019_core_corrected.csv
```

4. Abra `test.ipynb` para validação exploratória e QA dos dados processados.

## Entregáveis e Validação

### Artefatos de Saída Esperados

- `data/processed/pns_2019_core_corrected.csv`: Dataset reduzido (indivíduos 60+, colunas core + derivadas).
- `data/processed/pns_mappings.json`: Lookup de categorias e recodificações (ex.: autoav mapping).
- `test.ipynb`: Notebook com checagens, gráficos exploratórios e validação de estimativas ponderadas.

### Validação e QA

O `test.ipynb` contém checks de qualidade essenciais, incluindo:

- Distribuição de variáveis core.
- Taxas de missingness (antes e depois da imputação).
- Cross-tabs por UF/sexo/raça.
- Verificação de pesos amostrais.
- Correlação entre domínios (para verificar redundância).
- Comparação de estimativas ponderadas (ex: svymean de hipert) com tabelas agregadas oficiais do IBGE para validar o pipeline.

<details>
<summary><strong>Potenciais Análises e Insights</strong></summary>

- **Painel e Agregados**:
  - Índice AAI_Br total e por domínio, por município e UF.
  - Ranking de municípios (decis / percentis) — destacar 20% piores.
  - Decomposição do índice por domínio (heatmap).
  - Mapas choropleth: AAI_Br_total; domain heatmaps.

- **Perfis Populacionais**:
  - Prevalência de multimorbidade por faixa etária (60-69 / 70-79 / 80+) e por sexo/raça/area.
  - Taxa de limitação ADL/IADL por município.
  - Proporção de idosos dependentes do SUS vs. cobertura vacinal.

- **Modelagem e Associação**:
  - Regressões ponderadas (survey regression) para identificar determinantes de health_score e functional_score.
  - Modelos de classificação (ex.: Random Forest) para prever municípios com baixo AAI (feature importance / SHAP).

</details>

## Limitações e Próximos Passos

### Limitações Conhecidas

- A PNS mede uso e demanda percebida, não oferta de serviços (requer merge com DATASUS).
- Questionamentos sujeitos a viés de memória (recall bias) (ex.: consulta 12 meses).
- Agregações muito finas (ex: bairro) podem violar regras de divulgação (supressão).

### Planos Futuros (Roadmap)

- **Versão 2**: Incluir módulos de Imputação Múltipla (MICE) e modelagem robusta com survey design.
- **Integração**: Merge com DATASUS (SIH/SIA) e SISAB por codmun para medir oferta de serviços.
- **Benchmarking Temporal**: Incorporar PNS-2013 para análise de mudança 2013→2019.
- **Validação de Especialistas**: Rodada Delphi para definir pesos finais do índice.

## Contato e Citação

**Equipe**: [Lista de integrantes do grupo / Email de contato]

**Versão**: v0.1

**Data da Última Atualização**: YYYY-MM-DD

**Como Citar**: Documento técnico em preparação. Por enquanto, citar IBGE (PNS-2019) como fonte primária dos microdados.

## Apêndice: Pseudocódigo

Exemplo de fórmula para `health_score`:

```python
# 1. Padronizar componentes (Z-score)
# Assumindo que autoav 1=Péssimo, 5=Excelente (corrigir se necessário)
autoav_z = (autoav - mean(autoav)) / sd(autoav)
multimorb_z = (multimorbidity_count - mean(multimorbidity_count)) / sd(multimorbidity_count)
functional_z = (functional_score - mean(functional_score)) / sd(functional_score)

# 2. Score bruto ponderado (pesos w1, w2, w3 a validar)
# Multiplicar por -1 componentes onde "maior" é "pior"
# Se autoav (5=Excelente) e functional_score (1=Melhor) já estão na direção correta:
health_score_raw = w1 * (autoav_z) + w2 * (-multimorb_z) + w3 * (functional_z)

# 3. Normalizar para 0-1 (Min-Max Scaling)
health_score = (health_score_raw - min(health_score_raw)) / (max(health_score_raw) - min(health_score_raw))
```