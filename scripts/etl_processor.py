import os
import re
import glob
import pandas as pd
from sqlalchemy import text
from .db_config import get_postgres_engine

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASCII_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'ASCII')

BODY_SYSTEM_KEYWORDS = {
    'Cardiovascular': [
        'cardiac', 'heart', 'arrhythmia', 'myocardial', 'hypertension',
        'hypotension', 'stroke', 'thrombosis', 'angina', 'palpitation',
        'tachycardia', 'infarction',
    ],
    'Gastrointestinal': [
        'nausea', 'vomiting', 'diarrhea', 'diarrhoea', 'abdominal',
        'gastric', 'liver', 'pancreatitis', 'colitis', 'dyspepsia',
    ],
    'Neurological': [
        'headache', 'dizziness', 'seizure', 'tremor', 'confusion',
        'syncope', 'migraine', 'insomnia', 'depression', 'anxiety',
    ],
    'Respiratory': [
        'dyspnea', 'dyspnoea', 'cough', 'pneumonia', 'asthma',
        'respiratory', 'pulmonary', 'hypoxia',
    ],
    'Dermatological': [
        'rash', 'pruritus', 'urticaria', 'erythema', 'dermatitis',
        'skin', 'sweating', 'angioedema',
    ],
    'Musculoskeletal': [
        'arthralgia', 'myalgia', 'arthritis', 'muscle', 'joint',
        'back pain', 'fracture',
    ],
    'Renal': [
        'renal', 'kidney', 'creatinine', 'urinary', 'nephropathy',
    ],
}

OUTCOME_SEVERITY = {
    'DE': ('Death', 5),
    'LT': ('Life-threatening', 4),
    'HO': ('Hospitalization', 3),
    'DS': ('Disability', 2),
    'CA': ('Congenital anomaly', 2),
    'RI': ('Required intervention', 2),
    'OT': ('Other', 1),
}

_DOSE_PATTERN = re.compile(
    r'\s*\d+\.?\d*\s*(MG|MCG|ML|G|IU|UNITS?|%|TABLET|CAPSULE|PATCH|INJECTION|SOLUTION|INFUSION)\b.*',
    re.IGNORECASE,
)
_NONALPHA_PATTERN = re.compile(r'[^A-Z0-9\s\-]')


def upsert_ignore(table, conn, keys, data_iter):
    from sqlalchemy.dialects.postgresql import insert
    rows = [dict(zip(keys, row)) for row in data_iter]
    stmt = insert(table.table).values(rows).on_conflict_do_nothing()
    conn.execute(stmt)


def get_latest_file(pattern, quarter=None):
    files = glob.glob(os.path.join(ASCII_DIR, pattern))
    if quarter:
        files = [f for f in files if quarter in f]
    return sorted(files)[-1] if files else None


def normalize_drug_name(name):
    if pd.isna(name):
        return name
    name = str(name).upper().strip()
    name = _DOSE_PATTERN.sub('', name).strip()
    name = _NONALPHA_PATTERN.sub('', name).strip()
    return name or None


def map_body_system(reaction_term):
    if pd.isna(reaction_term):
        return 'Other'
    term_lower = str(reaction_term).lower()
    for system, keywords in BODY_SYSTEM_KEYWORDS.items():
        for kw in keywords:
            if kw in term_lower:
                return system
    return 'Other'


def run_etl(testing=True, quarter=None):
    engine = get_postgres_engine()
    nrows = 10000 if testing else None
    reports_df = None

    demo_file = get_latest_file("DEMO*.txt", quarter)
    drug_file = get_latest_file("DRUG*.txt", quarter)
    reac_file = get_latest_file("REAC*.txt", quarter)
    outc_file = get_latest_file("OUTC*.txt", quarter)

    if demo_file:
        print(f"loading {os.path.basename(demo_file)}")
        demo_df = pd.read_csv(demo_file, sep='$', dtype=str, encoding='latin1', nrows=nrows, low_memory=False, on_bad_lines='skip')

        col_map = {
            'primaryid':        'report_id',
            'event_dt':         'report_date',
            'age':              'age',
            'sex':              'sex',
            'reporter_country': 'country',
        }
        reports_df = (
            demo_df.rename(columns=col_map)
            .reindex(columns=list(col_map.values()))
            .copy()
        )
        reports_df['age'] = pd.to_numeric(reports_df['age'], errors='coerce')
        reports_df['report_date'] = pd.to_datetime(
            reports_df['report_date'], format='%Y%m%d', errors='coerce'
        )
        reports_df['drug_id'] = None
        reports_df.drop_duplicates(subset=['report_id'], inplace=True)
        reports_df.dropna(subset=['report_id'], inplace=True)

        print(f"  -> {len(reports_df):,} reports")
        reports_df[['report_id', 'drug_id', 'report_date', 'age', 'sex', 'country']].to_sql(
            'reports', engine, if_exists='append', index=False, chunksize=10000, method=upsert_ignore
        )

    if drug_file:
        print(f"loading {os.path.basename(drug_file)}")
        drug_df = pd.read_csv(drug_file, sep='$', dtype=str, encoding='latin1', nrows=nrows, low_memory=False, on_bad_lines='skip')

        drug_df['drugname'] = drug_df['drugname'].apply(normalize_drug_name)
        drug_df['drugname'] = drug_df['drugname'].str[:490]
        drug_df['prod_ai'] = drug_df['prod_ai'].str[:490]
        drug_df.dropna(subset=['drugname'], inplace=True)

        unique_drugs = (
            drug_df[['drugname', 'prod_ai']]
            .drop_duplicates()
            .rename(columns={'drugname': 'brand_name', 'prod_ai': 'generic_name'})
            .reset_index(drop=True)
        )
        print(f"  -> {len(unique_drugs):,} unique drugs")
        unique_drugs.to_sql('drugs', engine, if_exists='append', index=False, chunksize=10000, method=upsert_ignore)

        role_col = 'role_cod' if 'role_cod' in drug_df.columns else None
        if role_col:
            ps_drugs = drug_df[drug_df[role_col] == 'PS'][['primaryid', 'drugname']].drop_duplicates('primaryid')
        else:
            ps_drugs = drug_df[['primaryid', 'drugname']].drop_duplicates('primaryid')

        with engine.connect() as conn:
            db_drugs = pd.read_sql("SELECT drug_id, brand_name FROM drugs", conn)

        ps_merged = ps_drugs.merge(db_drugs, left_on='drugname', right_on='brand_name', how='left')
        ps_merged = ps_merged.dropna(subset=['drug_id'])[['primaryid', 'drug_id']]
        ps_merged.columns = ['report_id', 'drug_id']
        ps_merged['drug_id'] = ps_merged['drug_id'].astype(int)

        if not ps_merged.empty:
            print(f"  linking {len(ps_merged):,} reports to drugs...")
            with engine.begin() as conn:
                conn.execute(text(
                    "CREATE TEMP TABLE _drug_id_map (report_id VARCHAR(50), drug_id INTEGER) ON COMMIT DROP"
                ))
                ps_merged.to_sql('_drug_id_map', conn, if_exists='append', index=False)
                conn.execute(text("""
                    UPDATE reports r
                       SET drug_id = m.drug_id
                      FROM _drug_id_map m
                     WHERE r.report_id = m.report_id
                       AND r.drug_id IS NULL
                """))

    if reac_file and reports_df is not None:
        print(f"loading {os.path.basename(reac_file)}")
        reac_df = pd.read_csv(
            reac_file, sep='$', dtype=str, encoding='latin1',
            nrows=(nrows * 2 if nrows else None),
            low_memory=False, on_bad_lines='skip'
        )

        reac_db_df = (
            reac_df.rename(columns={'primaryid': 'report_id', 'pt': 'reaction_term'})
            [['report_id', 'reaction_term']]
            .copy()
        )
        reac_db_df['reaction_term'] = reac_db_df['reaction_term'].str.lower().str.strip()
        reac_db_df.drop_duplicates(subset=['report_id', 'reaction_term'], inplace=True)

        reac_db_df['body_system'] = reac_db_df['reaction_term'].apply(map_body_system)
        reac_db_df['severity'] = 'Standard'

        valid_ids = set(reports_df['report_id'].tolist())
        reac_db_df = reac_db_df[reac_db_df['report_id'].isin(valid_ids)]

        print(f"  -> {len(reac_db_df):,} reactions")
        reac_db_df.to_sql('reactions', engine, if_exists='append', index=False, chunksize=10000, method=upsert_ignore)

    if outc_file:
        print(f"loading {os.path.basename(outc_file)}")
        outc_df = pd.read_csv(outc_file, sep='$', dtype=str, encoding='latin1', nrows=nrows, low_memory=False, on_bad_lines='skip')

        outc_db_df = (
            outc_df.rename(columns={'primaryid': 'report_id', 'outc_cod': 'outcome_type'})
            [['report_id', 'outcome_type']]
            .copy()
        )
        outc_db_df.dropna(subset=['outcome_type'], inplace=True)
        outc_db_df.drop_duplicates(inplace=True)

        if reports_df is not None:
            valid_ids = set(reports_df['report_id'].tolist())
            outc_db_df = outc_db_df[outc_db_df['report_id'].isin(valid_ids)]

        print(f"  -> {len(outc_db_df):,} outcomes")
        outc_db_df.to_sql('outcomes', engine, if_exists='append', index=False, chunksize=10000, method=upsert_ignore)

        outc_db_df['severity_label'] = outc_db_df['outcome_type'].map(
            {k: v[0] for k, v in OUTCOME_SEVERITY.items()}
        ).fillna('Other')
        outc_db_df['severity_score'] = outc_db_df['outcome_type'].map(
            {k: v[1] for k, v in OUTCOME_SEVERITY.items()}
        ).fillna(1)

        report_severity = (
            outc_db_df.sort_values('severity_score', ascending=False)
            .drop_duplicates(subset=['report_id'])
            [['report_id', 'severity_label']]
        )

        print(f"  backfilling severity on {len(report_severity):,} reactions...")
        with engine.begin() as conn:
            conn.execute(text(
                "CREATE TEMP TABLE _sev_map (report_id VARCHAR(50), severity_label VARCHAR(50)) ON COMMIT DROP"
            ))
            report_severity.to_sql('_sev_map', conn, if_exists='append', index=False)
            conn.execute(text("""
                UPDATE reactions r
                   SET severity = m.severity_label
                  FROM _sev_map m
                 WHERE r.report_id = m.report_id
            """))

    print(f"ETL complete for quarter: {quarter or 'latest'}")


if __name__ == "__main__":
    for q in ['25Q1', '25Q2', '25Q3', '25Q4']:
        print(f"\n=== Running ETL for {q} ===")
        run_etl(testing=True, quarter=q)