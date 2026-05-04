import os
from sqlalchemy import create_engine
from pymongo import MongoClient

POSTGRES_USER = os.getenv("POSTGRES_USER", "faers_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "faers_password")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "faers_db")

POSTGRES_URI = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

MONGO_USER = os.getenv("MONGO_USER", "mongo_admin")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD", "mongo_password")
MONGO_HOST = os.getenv("MONGO_HOST", "localhost")
MONGO_PORT = os.getenv("MONGO_PORT", "27017")

MONGO_URI = f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"

def get_postgres_engine():
    return create_engine(POSTGRES_URI)

def get_mongo_client():
    return MongoClient(MONGO_URI)

def init_postgres_schema(engine):
    schema_sql = """
    CREATE TABLE IF NOT EXISTS drugs (
        drug_id SERIAL PRIMARY KEY,
        generic_name VARCHAR(500),
        brand_name VARCHAR(500)
    );

    CREATE TABLE IF NOT EXISTS reports (
        report_id VARCHAR(50) PRIMARY KEY,
        drug_id INTEGER REFERENCES drugs(drug_id),
        report_date DATE,
        age NUMERIC,
        sex VARCHAR(10),
        country VARCHAR(50)
    );

    CREATE TABLE IF NOT EXISTS reactions (
        reaction_id SERIAL PRIMARY KEY,
        report_id VARCHAR(50) REFERENCES reports(report_id),
        reaction_term VARCHAR(255),
        body_system VARCHAR(255),
        severity VARCHAR(50)
    );

    CREATE TABLE IF NOT EXISTS outcomes (
        outcome_id SERIAL PRIMARY KEY,
        report_id VARCHAR(50) REFERENCES reports(report_id),
        outcome_type VARCHAR(50)
    );
    
    CREATE INDEX IF NOT EXISTS idx_drug_id ON reports(drug_id);
    CREATE INDEX IF NOT EXISTS idx_report_date ON reports(report_date);
    """
    from sqlalchemy import text
    with engine.connect() as conn:
        conn.execute(text(schema_sql))
        conn.commit()

if __name__ == "__main__":
    print("setting up postgres schema...")
    engine = get_postgres_engine()
    init_postgres_schema(engine)
    print("done")
    
    print(f"testing mongo connection ({MONGO_URI})...")
    client = get_mongo_client()
    print("connected")

