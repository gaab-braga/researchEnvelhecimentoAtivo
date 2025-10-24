# ETL starter script for PNS (phase-1): select core columns, create derived indicators, and save a reduced CSV.
# This script is a template. Edit `COLUMN_MAP` to match the exact column names in your PNS microdata file.
# If the expected microdata file isn't present at `/mnt/data/pns_microdados_2019.csv`, the script will create
# a small synthetic example for demonstration.

import os
import pandas as pd
import numpy as np
import re

# Caminhos
RAW_FILE = "../data/raw/PNS_2019.txt"
SAS_FILE = "../data/raw/input_PNS_2019.sas"
OUTPUT_CSV = "../data/processed/pns_2019_pandas.csv"

# Colunas desejadas com códigos
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

def parse_sas_positions(sas_file):
    """Parse SAS file to get positions"""
    positions = {}
    pattern = re.compile(r'@(\d{5})\s*(\w+)\s*(\$?\d+(?:\.\d*)?)')
    with open(sas_file, 'r', encoding='latin-1') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                pos = int(match.group(1)) - 1  # 0-based
                code = match.group(2)
                length_str = match.group(3).replace('$', '').split('.')[0]
                length = int(length_str)
                positions[code] = (pos, length)
    return positions

def load_and_extract(raw_file, positions, desired_columns):
    """Load raw file and extract columns"""
    print("OK Carregando arquivo raw...")
    with open(raw_file, 'r', encoding='latin-1') as f:
        lines = f.readlines()
    
    data = {}
    for code, (pos, length) in positions.items():
        if code in desired_columns:
            col_name = desired_columns[code]
            data[col_name] = [line[pos:pos+length].strip() for line in lines]
    
    df = pd.DataFrame(data)
    print(f"OK Extraido: {df.shape}")
    return df

# Executar
positions = parse_sas_positions(SAS_FILE)
# Corrigir lengths erradas
positions["P00402"] = (602, 3)  # altura: 3 dígitos
positions["V0029"] = (1381, 1)  # area_urbana: 1 dígito
positions["V00293"] = (1503, 1)  # regiao: 1 dígito
print("Posições encontradas para desejadas:")
for code in DESIRED_COLUMNS:
    if code in positions:
        print(f"{code}: pos={positions[code][0]}, len={positions[code][1]}")
    else:
        print(f"{code}: não encontrado")
df = load_and_extract(RAW_FILE, positions, DESIRED_COLUMNS)

# Converter tipos
numeric_cols = ["peso_amostral", "idade", "anos_estudo", "renda_percapita", "num_medicamentos", "peso_real", "altura", "autoavaliacao_saude"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Filtro 60+
df = df[df["idade"] >= 60].copy()

# Calcular IMC
if "peso_real" in df.columns and "altura" in df.columns:
    df["altura"] = df["altura"] / 100  # assumir cm
    df["imc"] = df["peso_real"] / (df["altura"] ** 2)
    df["imc"] = df["imc"].fillna(df["imc"].median())

# Coerce binary
binary_cols = ["possui_plano_saude", "consulta_12m", "atendimento_sus", "usa_internet", "usa_celular", "depressao_diag", "vacina_influenza", "atividade_fisica", "fumante_atual", "hipertensao", "diabetes", "doenca_cardiaca", "avc", "doenca_respiratoria", "cancer"]
for col in binary_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

# Derived vars
chronic_cols = ["hipertensao", "diabetes", "doenca_cardiaca", "avc", "doenca_respiratoria", "cancer", "depressao_diag"]
if all(c in df.columns for c in chronic_cols):
    df["multimorbidity_count"] = df[chronic_cols].sum(axis=1)
    df["multimorb_cat"] = pd.cut(df["multimorbidity_count"], bins=[-1,0,1,2,100], labels=["0","1","2","3+"])

adl_cols = ["dificuldade_vestir", "dificuldade_banho", "dificuldade_alimentar"]
iadl_cols = ["dificuldade_compras", "dificuldade_medico"]
# Convert to numeric for summing
for col in adl_cols + iadl_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

if all(c in df.columns for c in adl_cols):
    df["adl_score"] = df[adl_cols].sum(axis=1)
if all(c in df.columns for c in iadl_cols):
    df["iadl_score"] = df[iadl_cols].sum(axis=1)

if "adl_score" in df.columns or "iadl_score" in df.columns:
    df["functional_raw"] = 0
    if "adl_score" in df.columns:
        df["functional_raw"] += df["adl_score"]
    if "iadl_score" in df.columns:
        df["functional_raw"] += df["iadl_score"]
    max_raw = df["functional_raw"].max() or 1
    df["functional_score"] = 1 - (df["functional_raw"] / max_raw)

if "possui_plano_saude" in df.columns and "atendimento_sus" in df.columns:
    df["dependencia_SUS"] = ((df["possui_plano_saude"] == 2) & (df["atendimento_sus"] == 1)).astype(int)

if "vacina_influenza" in df.columns:
    df["cobertura_influenza"] = df["vacina_influenza"]

# Health score
if "autoavaliacao_saude" in df.columns and "multimorbidity_count" in df.columns and "functional_score" in df.columns:
    df["autoav_z"] = (df["autoavaliacao_saude"].astype(float) - df["autoavaliacao_saude"].mean()) / df["autoavaliacao_saude"].std(ddof=0)
    df["multimorb_z"] = (df["multimorbidity_count"].astype(float) - df["multimorbidity_count"].mean()) / df["multimorbidity_count"].std(ddof=0)
    df["functional_z"] = (df["functional_score"].astype(float) - df["functional_score"].mean()) / df["functional_score"].std(ddof=0)
    df["health_score_raw"] = (-0.5 * df["autoav_z"]) + (-0.7 * df["multimorb_z"]) + (1.2 * df["functional_z"])
    minv = df["health_score_raw"].min()
    maxv = df["health_score_raw"].max()
    df["health_score"] = (df["health_score_raw"] - minv) / (maxv - minv + 1e-9)

# Impute
impute_cols = ["num_medicamentos", "idade", "anos_estudo", "renda_percapita"]
for col in impute_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# Mapeamentos categóricos
MAPPINGS = {
    "uf": {
        "11": "Rondônia", "12": "Acre", "13": "Amazonas", "14": "Roraima", "15": "Pará", "16": "Amapá", "17": "Tocantins",
        "21": "Maranhão", "22": "Piauí", "23": "Ceará", "24": "Rio Grande do Norte", "25": "Paraíba", "26": "Pernambuco", "27": "Alagoas", "28": "Sergipe", "29": "Bahia",
        "31": "Minas Gerais", "32": "Espírito Santo", "33": "Rio de Janeiro", "35": "São Paulo",
        "41": "Paraná", "42": "Santa Catarina", "43": "Rio Grande do Sul",
        "50": "Mato Grosso do Sul", "51": "Mato Grosso", "52": "Goiás", "53": "Distrito Federal"
    },
    "sexo": {"1": "Masculino", "2": "Feminino"},
    "raca_cor": {"1": "Branca", "2": "Preta", "3": "Amarela", "4": "Parda", "5": "Indígena", "9": "Ignorado"},
    "situacao_ocupacional": {"1": "Ocupado", "2": "Desocupado", "3": "Fora da força de trabalho", "4": "Não aplicável", "9": "Ignorado"},
    "area_urbana": {"1": "Urbano", "2": "Rural"},
    "regiao": {"1": "Norte", "2": "Nordeste", "3": "Sudeste", "4": "Sul", "5": "Centro-Oeste"},
    "autoavaliacao_saude": {"1.0": "Muito boa", "2.0": "Boa", "3.0": "Regular", "4.0": "Ruim", "5.0": "Muito ruim"},
    "hipertensao": {"0": "Não sabe", "1": "Sim", "2": "Não", "9": "Ignorado"},
    "diabetes": {"0": "Não sabe", "1": "Sim", "2": "Não", "9": "Ignorado"},
    "doenca_cardiaca": {"0": "Não sabe", "1": "Sim", "2": "Não", "9": "Ignorado"},
    "avc": {"0": "Não sabe", "1": "Sim", "2": "Não", "9": "Ignorado"},
    "doenca_respiratoria": {"0": "Não sabe", "1": "Sim", "2": "Não", "9": "Ignorado"},
    "cancer": {"0": "Não sabe", "1": "Sim", "2": "Não", "3": "Não sabe", "9": "Ignorado"},
    "depressao_diag": {"0": "Não sabe", "1": "Sim", "2": "Não", "3": "Não sabe", "4": "Não sabe", "5": "Ignorado", "9": "Ignorado"},
    "possui_plano_saude": {"1": "Sim", "2": "Não", "9": "Ignorado"},
    "consulta_12m": {"1": "Sim", "2": "Não", "3": "Não se aplica", "4": "Não sabe", "5": "Ignorado"},
    "atendimento_sus": {"0": "Não", "1": "Sim", "2": "Não", "3": "Não sabe"},
    "vacina_influenza": {"0": "Não sabe", "1": "Sim", "2": "Não", "9": "Ignorado"},
    "atividade_fisica": {"0": "Não sabe", "1": "Sim", "2": "Não", "9": "Ignorado"},
    "fumante_atual": {"0": "Não", "1": "Sim", "2": "Não", "3": "Não, mas já fumou", "9": "Ignorado"},
    "usa_internet": {"0": "Não", "1": "Sim", "2": "Não", "3": "Não sabe", "9": "Ignorado"},
    "usa_celular": {"0": "Não", "1": "Sim", "2": "Não", "9": "Ignorado"},
    "mora_sozinho": {"1": "Pessoa de referência", "2": "Cônjuge ou companheiro(a)", "3": "Filho(a)", "4": "Outro parente", "5": "Agregado", "6": "Empregado doméstico", "7": "Parente do empregado", "8": "Outro morador", "9": "Ignorado"},
    "dificuldade_alimentar": {"1": "Nenhuma dificuldade", "2": "Alguma dificuldade", "3": "Muita dificuldade", "4": "Não consegue de modo algum", "9": "Ignorado"},
    "dificuldade_banho": {"1": "Nenhuma dificuldade", "2": "Alguma dificuldade", "3": "Muita dificuldade", "4": "Não consegue de modo algum", "9": "Ignorado"},
    "dificuldade_vestir": {"1": "Nenhuma dificuldade", "2": "Alguma dificuldade", "3": "Muita dificuldade", "4": "Não consegue de modo algum", "9": "Ignorado"},
    "ajuda_adl": {"1": "Sim", "2": "Não", "9": "Ignorado"},
    "dificuldade_compras": {"1": "Nenhuma dificuldade", "2": "Alguma dificuldade", "3": "Muita dificuldade", "4": "Não consegue de modo algum", "9": "Ignorado"},
    "dificuldade_medico": {"1": "Nenhuma dificuldade", "2": "Alguma dificuldade", "3": "Muita dificuldade", "4": "Não consegue de modo algum", "9": "Ignorado"},
    "ajuda_iadl": {"1": "Sim", "2": "Não", "9": "Ignorado"},
    "queda_12m": {"1": "Sim", "2": "Não", "9": "Ignorado"},
    # Adicionar mais conforme necessário
}

# Aplicar mapeamentos
for col, mapping in MAPPINGS.items():
    if col in df.columns:
        print(f"OK Mapeando {col}: valores unicos antes: {df[col].unique()[:5]}")
        df[col] = df[col].astype(str).str.strip().str.lstrip('0').replace('', '0').map(mapping).fillna(df[col])  # Converte para string, remove zeros à esquerda e map
        print(f"Valores unicos depois: {df[col].unique()[:5]}")

# Salvar CSV
df.to_csv(OUTPUT_CSV, index=False)
print(f"OK Arquivo salvo em {OUTPUT_CSV}")
