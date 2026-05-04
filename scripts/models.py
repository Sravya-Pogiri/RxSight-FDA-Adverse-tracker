import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from .db_config import get_postgres_engine


def get_reaction_trends(drug_name):
    import requests

    db_counts = {}
    api_counts = {}

    # pull from DB
    try:
        from sqlalchemy import text
        engine = get_postgres_engine()
        query = text("""
            SELECT date_trunc('quarter', r.report_date) AS quarter_end_date,
                   count(*) AS report_count
              FROM reports r
              JOIN drugs d ON r.drug_id = d.drug_id
             WHERE d.brand_name = :drug_name
               AND r.report_date IS NOT NULL
             GROUP BY date_trunc('quarter', r.report_date)
             ORDER BY quarter_end_date ASC
        """)
        df = pd.read_sql(query, engine, params={'drug_name': drug_name})
        for _, row in df.iterrows():
            key = pd.Timestamp(row['quarter_end_date']).normalize()
            db_counts[key] = int(row['report_count'])
        print(f"DB returned {len(db_counts)} quarters for {drug_name}")
    except Exception as e:
        print(f"DB trend query failed ({e})")

    # pull from OpenFDA API
    try:
        quarters = [
            ('2020Q1', '20200101', '20200331'),
            ('2020Q2', '20200401', '20200630'),
            ('2020Q3', '20200701', '20200930'),
            ('2020Q4', '20201001', '20201231'),
            ('2021Q1', '20210101', '20210331'),
            ('2021Q2', '20210401', '20210630'),
            ('2021Q3', '20210701', '20210930'),
            ('2021Q4', '20211001', '20211231'),
            ('2022Q1', '20220101', '20220331'),
            ('2022Q2', '20220401', '20220630'),
            ('2022Q3', '20220701', '20220930'),
            ('2022Q4', '20221001', '20221231'),
            ('2023Q1', '20230101', '20230331'),
            ('2023Q2', '20230401', '20230630'),
            ('2023Q3', '20230701', '20230930'),
            ('2023Q4', '20231001', '20231231'),
            ('2024Q1', '20240101', '20240331'),
            ('2024Q2', '20240401', '20240630'),
            ('2024Q3', '20240701', '20240930'),
            ('2024Q4', '20241001', '20241231'),
            ('2025Q1', '20250101', '20250331'),
            ('2025Q2', '20250401', '20250630'),
            ('2025Q3', '20250701', '20250930'),
            ('2025Q4', '20251001', '20251231'),
        ]
        for label, start, end in quarters:
            r = requests.get(
                "https://api.fda.gov/drug/event.json",
                params={
                    'search': f'patient.drug.medicinalproduct:"{drug_name}" AND receivedate:[{start} TO {end}]',
                    'limit': 1
                },
                timeout=10
            )
            if r.status_code == 200:
                count = r.json().get('meta', {}).get('results', {}).get('total', 0)
                key = pd.Timestamp(end).normalize()
                api_counts[key] = count
        print(f"API returned {len(api_counts)} quarters for {drug_name}")
    except Exception as e:
        print(f"OpenFDA trend fetch failed ({e})")

    # merge - use max of DB and API for each quarter to avoid double counting
    # normalize all keys to timezone-naive
    db_counts = {k.tz_localize(None) if k.tzinfo else k: v for k, v in db_counts.items()}
    api_counts = {k.tz_localize(None) if k.tzinfo else k: v for k, v in api_counts.items()}

    all_keys = set(db_counts.keys()) | set(api_counts.keys())
    merged = {}
    for key in all_keys:
        db_val = db_counts.get(key, 0)
        api_val = api_counts.get(key, 0)
        merged[key] = max(db_val, api_val)

    if len(merged) >= 4:
        combined_df = pd.DataFrame([
            {'quarter_end_date': k, 'report_count': v}
            for k, v in sorted(merged.items())
        ])
        return combined_df

    # final fallback
    print("not enough data, using sample")
    dates = pd.date_range(start='2020-01-01', periods=12, freq='QE')
    counts = [100, 150, 130, 180, 210, 205, 250, 280, 310, 290, 350, 400]
    return pd.DataFrame({'quarter_end_date': dates, 'report_count': counts})


def train_trend_regression(df, test_size=0.2):
    from sklearn.tree import DecisionTreeRegressor

    if df is None or df.empty or len(df) < 4:
        return None, {}, pd.DataFrame()

    df = df.copy().sort_values('quarter_end_date').reset_index(drop=True)

    start_date = df['quarter_end_date'].min()
    df['days'] = (df['quarter_end_date'] - start_date).dt.days

    split_idx = max(1, int(len(df) * 0.8))
    train_df = df.iloc[:split_idx]
    test_df  = df.iloc[split_idx:]

    X_train = train_df[['days']]
    y_train = train_df['report_count']
    X_test  = test_df[['days']]
    y_test  = test_df['report_count']

    # --- Model 1: Linear Regression (original) ---
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    lr_train_pred = lr_model.predict(X_train)
    lr_r2_train = r2_score(y_train, lr_train_pred)

    lr_stats = {
        'r2_train': round(lr_r2_train, 3),
        'r2_test':  None,
        'rmse_test': None,
        'slope': round(float(lr_model.coef_[0]), 4),
    }
    if len(test_df) > 0:
        lr_test_pred = lr_model.predict(X_test)
        lr_stats['r2_test']   = round(r2_score(y_test, lr_test_pred), 3)
        lr_stats['rmse_test'] = round(float(np.sqrt(mean_squared_error(y_test, lr_test_pred))), 2)

    # --- Model 2: Decision Tree Regressor (improved) ---
    dt_model = DecisionTreeRegressor(max_depth=4, random_state=42)
    dt_model.fit(X_train, y_train)

    dt_train_pred = dt_model.predict(X_train)
    dt_r2_train = r2_score(y_train, dt_train_pred)

    dt_stats = {
        'r2_train': round(dt_r2_train, 3),
        'r2_test':  None,
        'rmse_test': None,
    }
    if len(test_df) > 0:
        dt_test_pred = dt_model.predict(X_test)
        dt_stats['r2_test']   = round(r2_score(y_test, dt_test_pred), 3)
        dt_stats['rmse_test'] = round(float(np.sqrt(mean_squared_error(y_test, dt_test_pred))), 2)

    # --- Build overlay_df with both model predictions ---
    overlay_rows = []
    last_day = int(df['days'].max())

    for _, row in train_df.iterrows():
        overlay_rows.append({
            'quarter_end_date': row['quarter_end_date'],
            'actual':           row['report_count'],
            'lr_fitted':        lr_model.predict([[row['days']]])[0],
            'dt_fitted':        dt_model.predict([[row['days']]])[0],
            'split':            'train',
        })
    for _, row in test_df.iterrows():
        overlay_rows.append({
            'quarter_end_date': row['quarter_end_date'],
            'actual':           row['report_count'],
            'lr_fitted':        lr_model.predict([[row['days']]])[0],
            'dt_fitted':        dt_model.predict([[row['days']]])[0],
            'split':            'test',
        })

    # Forecast: decision tree can't extrapolate beyond training range,
    # so we use linear regression for the 2-quarter forecast
    for i in range(1, 3):
        future_day  = last_day + i * 91
        future_date = start_date + pd.Timedelta(days=future_day)
        overlay_rows.append({
            'quarter_end_date': future_date,
            'actual':           None,
            'lr_fitted':        lr_model.predict([[future_day]])[0],
            'dt_fitted':        None,   # DT can't extrapolate
            'split':            'forecast',
        })

    overlay_df = pd.DataFrame(overlay_rows)

    combined_stats = {
        'lr': lr_stats,
        'dt': dt_stats,
    }
    return lr_model, combined_stats, overlay_df

def cluster_reactions(reactions_list, n_clusters=5):
    if not reactions_list or len(reactions_list) < 5:
        return pd.DataFrame()

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(reactions_list)

    k = min(n_clusters, len(reactions_list))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X.toarray())

    return pd.DataFrame({
        'reaction': reactions_list,
        'cluster':  [f'Cluster {l + 1}' for l in labels],
        'x':        coords[:, 0],
        'y':        coords[:, 1],
    })


def get_demographics(drug_name):
    try:
        from sqlalchemy import text
        engine = get_postgres_engine()
        age_query = text("""
            SELECT r.age
              FROM reports r
              JOIN drugs d ON r.drug_id = d.drug_id
             WHERE d.brand_name = :drug_name
               AND r.age IS NOT NULL
               AND r.age BETWEEN 0 AND 120
             LIMIT 5000
        """)
        sex_query = text("""
            SELECT r.sex, count(*) AS count
              FROM reports r
              JOIN drugs d ON r.drug_id = d.drug_id
             WHERE d.brand_name = :drug_name
             GROUP BY r.sex
        """)
        age_df = pd.read_sql(age_query, engine, params={'drug_name': drug_name})
        sex_df = pd.read_sql(sex_query, engine, params={'drug_name': drug_name})

        if not age_df.empty and not sex_df.empty:
            return age_df, sex_df
    except Exception as e:
        print(f"demographics query failed ({e}), using mock data")

    rng = np.random.default_rng(42)
    ages = rng.normal(loc=55, scale=18, size=300).clip(0, 100)
    age_df = pd.DataFrame({'age': ages})
    sex_df = pd.DataFrame({'sex': ['M', 'F', 'UNK'], 'count': [140, 120, 40]})
    return age_df, sex_df


def get_severity_breakdown(drug_name):
    try:
        from sqlalchemy import text
        engine = get_postgres_engine()
        query = text("""
            SELECT rx.severity, count(*) AS count
              FROM reactions rx
              JOIN reports r  ON rx.report_id = r.report_id
              JOIN drugs d    ON r.drug_id = d.drug_id
             WHERE d.brand_name = :drug_name
             GROUP BY rx.severity
             ORDER BY count DESC
        """)
        df = pd.read_sql(query, engine, params={'drug_name': drug_name})
        if not df.empty:
            return df
    except Exception as e:
        print(f"severity query failed ({e}), using mock data")

    return pd.DataFrame({
        'severity': ['Other', 'Hospitalization', 'Life-threatening', 'Death', 'Disability'],
        'count':    [220, 85, 40, 25, 15],
    })


if __name__ == "__main__":
    drug = "IBUPROFEN"
    print(f"=== Trend: {drug} ===")
    trends = get_reaction_trends(drug)
    model, stats, overlay = train_trend_regression(trends)
    print(stats)
    print(overlay.tail(4))

    print(f"\n=== Demographics: {drug} ===")
    age_df, sex_df = get_demographics(drug)
    print(age_df.describe())
    print(sex_df)

    print(f"\n=== Severity: {drug} ===")
    sev = get_severity_breakdown(drug)
    print(sev)

    print(f"\n=== Cluster test ===")
    test_reactions = [
        "cardiac arrest", "heart failure", "nausea", "vomiting", "headache",
        "dizziness", "arrhythmia", "rash", "fatigue", "anaemia",
    ]
    cluster_df = cluster_reactions(test_reactions)
    print(cluster_df)