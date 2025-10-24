#!/usr/bin/env python3
"""
ETL PNS 2019 - Versão PySpark Final
====================================
Baseado na versão Pandas funcionando, otimizado para processamento distribuído.
Mantém a mesma lógica de extração, transformação e derivação de variáveis.
"""

import os
import sys
import re
from pathlib import Path
import pandas as pd
import numpy as np

# ==========================================
# CONFIGURAÇÕES
# ==========================================

BASE_PATH = Path("c:/Users/gafeb/Downloads/PNS_2019_20220525")
RAW_FILE = BASE_PATH / "data" / "raw" / "PNS_2019.txt"
SAS_FILE = BASE_PATH / "data" / "raw" / "input_PNS_2019.sas"
OUTPUT_DIR = BASE_PATH / "pns_2019_processado"
OUTPUT_CSV = OUTPUT_DIR / "pns_2019_final_completo.csv"

# Criar diretório de saída
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# COLUNAS DESEJADAS (do código Pandas)
# ==========================================

DESIRED_COLUMNS = {
    "V0001": "uf",
    "V00291": "peso_amostral",
    "UPA_PNS": "upa",
    "V0024": "estrato",
    "V0015": "id_domicilio",
    "V0028": "id_individuo",
    "V0029": "area_urbana",
    "V0029A": "cod_mun_ibge",
    "V0031": "area_metropolitana",
    "V00293": "regiao",
    "C006": "sexo",
    "C008": "idade",
    "C009": "raca_cor",
    "D00901": "anos_estudo",
    "E01602": "renda_percapita",
    "C011": "situacao_ocupacional",
    "C004": "mora_sozinho",
    "V0026": "num_pessoas_domicilio",
    "P00102": "autoavaliacao_saude",
    "Q00201": "hipertensao",
    "Q03001": "diabetes",
    "Q060": "doenca_cardiaca",
    "Q06207": "avc",
    "Q092": "doenca_respiratoria",
    "Q11201": "cancer",
    "N001": "depressao_diag",
    "I00101": "possui_plano_saude",
    "J001": "consulta_12m",
    "J01101": "internacoes_12m",
    "O00101": "vacina_influenza",
    "R002010": "num_medicamentos",
    "P034": "atividade_fisica",
    "P050": "fumante_atual",
    "M001": "usa_internet",
    "M011011": "usa_celular",
    "K004": "dificuldade_banho",
    "K010": "dificuldade_vestir",
    "K001": "dificuldade_alimentar",
    "K01901": "ajuda_adl",
    "K022": "dificuldade_compras",
    "K031": "dificuldade_medico",
    "K03401": "ajuda_iadl",
    "K05401": "queda_12m",
    "J026": "atendimento_sus",
    "P00103": "peso_real",
    "P00402": "altura",
}

# Correções de posições (do código Pandas)
POSITION_CORRECTIONS = {
    "P00402": (602, 3),   # altura: 3 dígitos
    "V0029": (1381, 1),   # area_urbana: 1 dígito
    "V00293": (1503, 1),  # regiao: 1 dígito
}

# ==========================================
# MAPEAMENTOS CATEGÓRICOS (do código Pandas)
# ==========================================

MAPPINGS = {
    "uf": {
        "11": "Rondônia", "12": "Acre", "13": "Amazonas", "14": "Roraima", 
        "15": "Pará", "16": "Amapá", "17": "Tocantins",
        "21": "Maranhão", "22": "Piauí", "23": "Ceará", "24": "Rio Grande do Norte", 
        "25": "Paraíba", "26": "Pernambuco", "27": "Alagoas", "28": "Sergipe", "29": "Bahia",
        "31": "Minas Gerais", "32": "Espírito Santo", "33": "Rio de Janeiro", "35": "São Paulo",
        "41": "Paraná", "42": "Santa Catarina", "43": "Rio Grande do Sul",
        "50": "Mato Grosso do Sul", "51": "Mato Grosso", "52": "Goiás", "53": "Distrito Federal"
    },
    "sexo": {"1": "Masculino", "2": "Feminino"},
    "raca_cor": {"1": "Branca", "2": "Preta", "3": "Amarela", "4": "Parda", "5": "Indígena", "9": "Ignorado"},
    "situacao_ocupacional": {"1": "Ocupado", "2": "Desocupado", "3": "Fora da força de trabalho", "4": "Não aplicável", "9": "Ignorado"},
    "area_urbana": {"1": "Urbano", "2": "Rural"},
    "regiao": {"1": "Norte", "2": "Nordeste", "3": "Sudeste", "4": "Sul", "5": "Centro-Oeste"},
    "autoavaliacao_saude": {"1": "Muito boa", "2": "Boa", "3": "Regular", "4": "Ruim", "5": "Muito ruim"},
    "hipertensao": {"1": "Sim", "2": "Não", "0": "Não sabe", "9": "Ignorado"},
    "diabetes": {"1": "Sim", "2": "Não", "0": "Não sabe", "9": "Ignorado"},
    "doenca_cardiaca": {"1": "Sim", "2": "Não", "0": "Não sabe", "9": "Ignorado"},
    "avc": {"1": "Sim", "2": "Não", "0": "Não sabe", "9": "Ignorado"},
    "doenca_respiratoria": {"1": "Sim", "2": "Não", "0": "Não sabe", "9": "Ignorado"},
    "cancer": {"1": "Sim", "2": "Não", "0": "Não sabe", "3": "Não sabe", "9": "Ignorado"},
    "depressao_diag": {"1": "Sim", "2": "Não", "0": "Não sabe", "9": "Ignorado"},
    "possui_plano_saude": {"1": "Sim", "2": "Não", "9": "Ignorado"},
    "consulta_12m": {"1": "Sim", "2": "Não", "3": "Não se aplica", "9": "Ignorado"},
    "atendimento_sus": {"1": "Sim", "2": "Não", "0": "Não", "3": "Não sabe"},
    "vacina_influenza": {"1": "Sim", "2": "Não", "0": "Não sabe", "9": "Ignorado"},
    "atividade_fisica": {"1": "Sim", "2": "Não", "0": "Não sabe", "9": "Ignorado"},
    "fumante_atual": {"1": "Sim", "2": "Não", "3": "Não, mas já fumou", "0": "Não", "9": "Ignorado"},
    "usa_internet": {"1": "Sim", "2": "Não", "0": "Não", "3": "Não sabe", "9": "Ignorado"},
    "usa_celular": {"1": "Sim", "2": "Não", "0": "Não", "9": "Ignorado"},
    "dificuldade_alimentar": {"1": "Nenhuma", "2": "Alguma", "3": "Muita", "4": "Não consegue", "9": "Ignorado"},
    "dificuldade_banho": {"1": "Nenhuma", "2": "Alguma", "3": "Muita", "4": "Não consegue", "9": "Ignorado"},
    "dificuldade_vestir": {"1": "Nenhuma", "2": "Alguma", "3": "Muita", "4": "Não consegue", "9": "Ignorado"},
    "dificuldade_compras": {"1": "Nenhuma", "2": "Alguma", "3": "Muita", "4": "Não consegue", "9": "Ignorado"},
    "dificuldade_medico": {"1": "Nenhuma", "2": "Alguma", "3": "Muita", "4": "Não consegue", "9": "Ignorado"},
    "queda_12m": {"1": "Sim", "2": "Não", "9": "Ignorado"},
    "ajuda_adl": {"1": "Sim", "2": "Não", "9": "Ignorado"},
    "ajuda_iadl": {"1": "Sim", "2": "Não", "9": "Ignorado"},
}

# ==========================================
# SETUP SPARK
# ==========================================

print("="*70)
print("🚀 ETL PNS 2019 - VERSÃO PYSPARK FINAL")
print("="*70)

# Configurar ambiente Windows
os.environ['HADOOP_HOME'] = ''
os.environ['hadoop.home.dir'] = ''
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Imports PySpark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, substring, when, lit, trim, regexp_replace, length as spark_length
from pyspark.sql.types import DoubleType, IntegerType, StringType
import findspark

findspark.init()

# Criar Spark
spark = SparkSession.builder \
    .appName("ETL_PNS_2019_Final") \
    .master("local[*]") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.sql.shuffle.partitions", "32") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print(f"✅ Spark {spark.version} iniciado\n")

# ==========================================
# FUNÇÕES AUXILIARES
# ==========================================

def parse_sas_positions(sas_file):
    """Parse SAS file para obter posições (0-based)"""
    positions = {}
    pattern = re.compile(r'@(\d{5})\s*(\w+)\s*(\$?\d+(?:\.\d*)?)')
    
    with open(sas_file, 'r', encoding='latin-1') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                pos = int(match.group(1)) - 1  # 0-based como no Pandas
                code = match.group(2)
                length_str = match.group(3).replace('$', '').split('.')[0]
                length = int(length_str)
                positions[code] = (pos, length)
    
    # Aplicar correções
    positions.update(POSITION_CORRECTIONS)
    
    return positions

def clean_and_map_column(df, col_name, mapping):
    """
    Limpa coluna (remove zeros à esquerda, trim) e aplica mapeamento.
    Replica lógica do Pandas: .strip().lstrip('0').replace('', '0')
    """
    # Trim
    df = df.withColumn(col_name, trim(col(col_name)))
    
    # Remover zeros à esquerda (mas manter '0' se for só zeros)
    df = df.withColumn(
        col_name,
        when(regexp_replace(col(col_name), "^0+", "") == "", "0")
        .otherwise(regexp_replace(col(col_name), "^0+", ""))
    )
    
    # Aplicar mapeamento
    mapping_expr = col(col_name)
    for key, value in mapping.items():
        mapping_expr = when(col(col_name) == key, value).otherwise(mapping_expr)
    
    df = df.withColumn(col_name, mapping_expr)
    return df

# ==========================================
# ETAPA 1: PARSE POSIÇÕES
# ==========================================

print("📖 [1/6] Parseando posições do SAS...")
positions = parse_sas_positions(SAS_FILE)

print(f"✅ {len(positions)} posições encontradas\n")
print("📋 Posições para colunas desejadas:")
for code in list(DESIRED_COLUMNS.keys())[:10]:
    if code in positions:
        print(f"   {code:15} → pos={positions[code][0]:4}, len={positions[code][1]}")
print(f"   ... e mais {len(DESIRED_COLUMNS) - 10}")

# ==========================================
# ETAPA 2: EXTRAÇÃO COM SPARK
# ==========================================

print("\n📥 [2/6] Extraindo colunas do arquivo raw...")

# Ler arquivo como texto
df_raw = spark.read.text(str(RAW_FILE), lineSep="\n")
n_total = df_raw.count()
print(f"   ✓ {n_total:,} registros lidos")

# Extrair apenas colunas desejadas
extract_expr = []
codes_found = []

for code, col_name in DESIRED_COLUMNS.items():
    if code in positions:
        pos, length = positions[code]
        # Substring é 1-based no Spark, então pos+1
        extract_expr.append(substring(col("value"), pos + 1, length).alias(col_name))
        codes_found.append(code)
    else:
        print(f"   ⚠️  {code} não encontrado no SAS")

df_extracted = df_raw.select(*extract_expr)
print(f"   ✓ {len(extract_expr)} colunas extraídas")

# ==========================================
# ETAPA 3: CONVERSÃO DE TIPOS
# ==========================================

print("\n🔢 [3/6] Convertendo tipos de dados...")

# Colunas numéricas
numeric_cols = [
    "peso_amostral", "idade", "anos_estudo", "renda_percapita", 
    "num_medicamentos", "peso_real", "altura", "autoavaliacao_saude",
    "internacoes_12m"
]

# Converter numéricas
for col_name in numeric_cols:
    if col_name in df_extracted.columns:
        df_extracted = df_extracted.withColumn(
            col_name,
            when(
                trim(col(col_name)).rlike("^[0-9]+$") & (trim(col(col_name)) != ""),
                col(col_name).cast(DoubleType())
            ).otherwise(lit(None))
        )

# Trim em todas as strings
string_cols = [c for c in df_extracted.columns if c not in numeric_cols]
for col_name in string_cols:
    df_extracted = df_extracted.withColumn(col_name, trim(col(col_name)))

print(f"   ✓ {len(numeric_cols)} colunas numéricas convertidas")

# ==========================================
# ETAPA 4: FILTRO 60+ ANOS
# ==========================================

print("\n🎯 [4/6] Filtrando população 60+ anos...")
df_filtered = df_extracted.filter(col("idade") >= 60)
n_60plus = df_filtered.count()
print(f"   ✓ {n_total:,} → {n_60plus:,} registros ({(n_60plus/n_total)*100:.1f}%)")

# ==========================================
# ETAPA 5: MAPEAMENTOS CATEGÓRICOS
# ==========================================

print("\n🏷️  [5/6] Aplicando mapeamentos categóricos...")

mapped_count = 0
for col_name, mapping in MAPPINGS.items():
    if col_name in df_filtered.columns:
        print(f"   Mapeando {col_name}...", end="")
        df_filtered = clean_and_map_column(df_filtered, col_name, mapping)
        mapped_count += 1
        print(" ✓")

print(f"   ✓ {mapped_count} colunas mapeadas")

# ==========================================
# ETAPA 6: VARIÁVEIS DERIVADAS
# ==========================================

print("\n🧮 [6/6] Criando variáveis derivadas...")

# Converter para Pandas para operações complexas
print("   Convertendo para Pandas...")
df_pandas = df_filtered.toPandas()

# 1. IMC
print("   [a] Calculando IMC...")
if "peso_real" in df_pandas.columns and "altura" in df_pandas.columns:
    df_pandas["altura_m"] = df_pandas["altura"] / 100  # cm para metros
    df_pandas["imc"] = df_pandas["peso_real"] / (df_pandas["altura_m"] ** 2)
    df_pandas["imc"] = df_pandas["imc"].fillna(df_pandas["imc"].median())
    print(f"      ✓ IMC calculado (média: {df_pandas['imc'].mean():.2f})")

# 2. Converter binários para 0/1
print("   [b] Convertendo variáveis binárias...")
binary_cols = [
    "possui_plano_saude", "consulta_12m", "atendimento_sus", "usa_internet",
    "usa_celular", "depressao_diag", "vacina_influenza", "atividade_fisica",
    "fumante_atual", "hipertensao", "diabetes", "doenca_cardiaca", "avc",
    "doenca_respiratoria", "cancer"
]

for col_name in binary_cols:
    if col_name in df_pandas.columns:
        # Converter 'Sim'/'Não' para 1/0
        df_pandas[col_name] = df_pandas[col_name].replace({"Sim": 1, "Não": 0, "Não sabe": 0, "Ignorado": 0})
        df_pandas[col_name] = pd.to_numeric(df_pandas[col_name], errors='coerce').fillna(0).astype(int)

# 3. Multimorbidade
print("   [c] Calculando multimorbidade...")
chronic_cols = ["hipertensao", "diabetes", "doenca_cardiaca", "avc", "doenca_respiratoria", "cancer", "depressao_diag"]
chronic_present = [c for c in chronic_cols if c in df_pandas.columns]

if chronic_present:
    df_pandas["multimorbidade_count"] = df_pandas[chronic_present].sum(axis=1)
    df_pandas["multimorbidade_cat"] = pd.cut(
        df_pandas["multimorbidade_count"],
        bins=[-1, 0, 1, 2, 100],
        labels=["0", "1", "2", "3+"]
    ).astype(str)
    print(f"      ✓ Multimorbidade (média: {df_pandas['multimorbidade_count'].mean():.2f})")

# 4. Escores funcionais (ADL/IADL)
print("   [d] Calculando escores funcionais...")
adl_cols = ["dificuldade_vestir", "dificuldade_banho", "dificuldade_alimentar"]
iadl_cols = ["dificuldade_compras", "dificuldade_medico"]

# Converter dificuldades para numérico (Nenhuma=0, Alguma=1, Muita=2, Não consegue=3)
difficulty_mapping = {"Nenhuma": 0, "Alguma": 1, "Muita": 2, "Não consegue": 3, "Ignorado": 0}

for col_name in adl_cols + iadl_cols:
    if col_name in df_pandas.columns:
        df_pandas[col_name] = df_pandas[col_name].replace(difficulty_mapping)
        df_pandas[col_name] = pd.to_numeric(df_pandas[col_name], errors='coerce').fillna(0).astype(int)

adl_present = [c for c in adl_cols if c in df_pandas.columns]
iadl_present = [c for c in iadl_cols if c in df_pandas.columns]

if adl_present:
    df_pandas["adl_score"] = df_pandas[adl_present].sum(axis=1)
    print(f"      ✓ ADL Score (média: {df_pandas['adl_score'].mean():.2f})")

if iadl_present:
    df_pandas["iadl_score"] = df_pandas[iadl_present].sum(axis=1)
    print(f"      ✓ IADL Score (média: {df_pandas['iadl_score'].mean():.2f})")

# 5. Functional Score normalizado
if adl_present or iadl_present:
    df_pandas["functional_raw"] = 0
    if "adl_score" in df_pandas.columns:
        df_pandas["functional_raw"] += df_pandas["adl_score"]
    if "iadl_score" in df_pandas.columns:
        df_pandas["functional_raw"] += df_pandas["iadl_score"]
    
    max_raw = df_pandas["functional_raw"].max() if df_pandas["functional_raw"].max() > 0 else 1
    df_pandas["functional_score"] = 1 - (df_pandas["functional_raw"] / max_raw)
    print(f"      ✓ Functional Score (média: {df_pandas['functional_score'].mean():.3f})")

# 6. Dependência SUS
print("   [e] Calculando dependência SUS...")
if "possui_plano_saude" in df_pandas.columns and "atendimento_sus" in df_pandas.columns:
    df_pandas["dependencia_SUS"] = (
        (df_pandas["possui_plano_saude"] == 0) & (df_pandas["atendimento_sus"] == 1)
    ).astype(int)
    print(f"      ✓ Dependência SUS ({df_pandas['dependencia_SUS'].sum():,} pessoas, {df_pandas['dependencia_SUS'].mean()*100:.1f}%)")

# 7. Cobertura influenza
if "vacina_influenza" in df_pandas.columns:
    df_pandas["cobertura_influenza"] = df_pandas["vacina_influenza"]

# 8. Health Score composto
print("   [f] Calculando Health Score...")
required_cols = ["autoavaliacao_saude", "multimorbidade_count", "functional_score"]
if all(c in df_pandas.columns for c in required_cols):
    # Converter autoavaliação para numérico
    health_mapping = {"Muito boa": 1, "Boa": 2, "Regular": 3, "Ruim": 4, "Muito ruim": 5}
    df_pandas["autoav_numeric"] = df_pandas["autoavaliacao_saude"].replace(health_mapping)
    df_pandas["autoav_numeric"] = pd.to_numeric(df_pandas["autoav_numeric"], errors='coerce')
    
    # Z-scores
    df_pandas["autoav_z"] = (df_pandas["autoav_numeric"] - df_pandas["autoav_numeric"].mean()) / df_pandas["autoav_numeric"].std(ddof=0)
    df_pandas["multimorb_z"] = (df_pandas["multimorbidade_count"] - df_pandas["multimorbidade_count"].mean()) / df_pandas["multimorbidade_count"].std(ddof=0)
    df_pandas["functional_z"] = (df_pandas["functional_score"] - df_pandas["functional_score"].mean()) / df_pandas["functional_score"].std(ddof=0)
    
    # Score composto
    df_pandas["health_score_raw"] = (
        (-0.5 * df_pandas["autoav_z"]) +
        (-0.7 * df_pandas["multimorb_z"]) +
        (1.2 * df_pandas["functional_z"])
    )
    
    # Normalizar 0-1
    min_val = df_pandas["health_score_raw"].min()
    max_val = df_pandas["health_score_raw"].max()
    df_pandas["health_score"] = (df_pandas["health_score_raw"] - min_val) / (max_val - min_val + 1e-9)
    print(f"      ✓ Health Score (média: {df_pandas['health_score'].mean():.3f})")

# 9. Imputação
print("   [g] Imputando valores faltantes...")
impute_cols = ["num_medicamentos", "idade", "anos_estudo", "renda_percapita"]
for col_name in impute_cols:
    if col_name in df_pandas.columns:
        missing = df_pandas[col_name].isnull().sum()
        if missing > 0:
            df_pandas[col_name] = df_pandas[col_name].fillna(df_pandas[col_name].median())
            print(f"      • {col_name}: {missing:,} valores imputados")

print("\n✅ Todas as variáveis derivadas criadas!")

# ==========================================
# EXPORTAÇÃO FINAL
# ==========================================

print("\n" + "="*70)
print("💾 EXPORTAÇÃO FINAL")
print("="*70)

# Salvar CSV
print(f"\n📝 Salvando {OUTPUT_CSV.name}...")
df_pandas.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
file_size = OUTPUT_CSV.stat().st_size / 1024**2

print(f"✅ Arquivo salvo com sucesso!")
print(f"   📁 Caminho: {OUTPUT_CSV}")
print(f"   📊 Registros: {len(df_pandas):,}")
print(f"   📋 Colunas: {len(df_pandas.columns)}")
print(f"   💾 Tamanho: {file_size:.1f} MB")

# Salvar metadados
metadata_file = OUTPUT_DIR / "metadados_completo.txt"
with open(metadata_file, 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("PNS 2019 - DATASET COMPLETO COM VARIÁVEIS DERIVADAS\n")
    f.write("="*70 + "\n\n")
    f.write(f"Registros: {len(df_pandas):,}\n")
    f.write(f"Colunas: {len(df_pandas.columns)}\n\n")
    f.write("COLUNAS:\n")
    f.write("-"*70 + "\n")
    for col in df_pandas.columns:
        f.write(f"{col:35} {df_pandas[col].dtype}\n")

print(f"\n📋 Metadados salvos: {metadata_file.name}")

# Resumo estatístico
print("\n📊 RESUMO ESTATÍSTICO:")
print("-"*70)

summary_vars = {
    "Idade": "idade",
    "Sexo": "sexo",
    "Multimorbidade": "multimorbidade_count",
    "Functional Score": "functional_score",
    "Health Score": "health_score",
    "Dependência SUS": "dependencia_SUS"
}

for label, var in summary_vars.items():
    if var in df_pandas.columns:
        if df_pandas[var].dtype in ['object', 'category']:
            top_value = df_pandas[var].value_counts().head(1)
            if len(top_value) > 0:
                print(f"{label:20} (categórica) - Mais comum: {top_value.index[0]} ({top_value.values[0]:,})")
        else:
            mean_val = df_pandas[var].mean()
            median_val = df_pandas[var].median()
            print(f"{label:20} (numérica)    - Média: {mean_val:.2f}, Mediana: {median_val:.2f}")

print("\n" + "="*70)
print("🎉 ETL CONCLUÍDO COM SUCESSO!")
print("="*70)
print("\n✨ Dataset pronto para análise!")

# Encerrar Spark
spark.stop()
print("\n🔌 Spark encerrado")