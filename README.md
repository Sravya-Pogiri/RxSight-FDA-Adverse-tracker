```markdown
# RxSight: FDA Adverse Event Tracker

RxSight is a web application that makes FDA adverse event data accessible to non-technical users. You can search any drug by name and see what side effects are being reported, how severe the outcomes are, whether reporting is trending up or down over time, and how reactions group by body system.

---

## What this does

The app downloads quarterly FAERS data from the FDA, cleans and loads it into PostgreSQL, and pulls live data from the OpenFDA API into MongoDB. It merges both sources to build a 24-quarter time series for trend forecasting, trains a linear regression and decision tree model on that data, uses K-Means clustering to group similar reaction terms, and displays everything in a Streamlit dashboard with 3 tabs.

---

## Requirements

- Python 3.9+
- Docker Desktop
- pip

---

## Setup

**Step 1: Clone the repo and set up virtual environment**

```bash
git clone <your-repo-url>
cd FDA-Adverse-tracker
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Step 2: Start the databases**

Docker is used to run PostgreSQL and MongoDB locally.

```bash
docker compose up -d
```

This starts:
- PostgreSQL on port 5432
- MongoDB on port 27017
- Adminer (database UI) on port 8080 — login at http://localhost:8080

**Step 3: Set up the database tables**

```bash
python -m scripts.db_config
```

**Step 4: Download the FAERS data**

```bash
python -m scripts.data_acquisition
```

This scrapes the FDA website, downloads the ZIP files for the most recent quarters, and unzips them into `data/raw/ASCII/`.

**Step 5: Run the ETL pipeline**

```bash
python -m scripts.etl_processor
```

Runs in test mode by default, which caps rows per file to keep things manageable. The pipeline uses upsert logic so it is safe to rerun if it crashes partway through — it picks up where it left off.

Note: The FAERS files are large. On macOS you may occasionally see a TimeoutError (errno 60) mid-read. This is a known issue with large file reads in pandas on macOS. Just rerun the script and progress will not be lost.

**Step 6: Launch the dashboard**

```bash
streamlit run app/main.py
```

Open http://localhost:8501 in your browser.

---

## Loading Sample Data (Optional)

If you want to verify the dashboard works without running the full ETL, sample data files are included in the `data/` folder. Load them with:

```bash
docker exec -it faers_postgres psql -U faers_user -d faers_db -c "\COPY drugs FROM '/sample_drugs.csv' CSV HEADER;"
docker exec -it faers_postgres psql -U faers_user -d faers_db -c "\COPY reports FROM '/sample_reports.csv' CSV HEADER;"
docker exec -it faers_postgres psql -U faers_user -d faers_db -c "\COPY reactions FROM '/sample_reactions.csv' CSV HEADER;"
docker exec -it faers_postgres psql -U faers_user -d faers_db -c "\COPY outcomes FROM '/sample_outcomes.csv' CSV HEADER;"
```

---

## Dashboard Tabs

**Tab 1 - Reactions and Profile**
Search a drug name to see the top 15 reported adverse reactions, a severity breakdown by outcome type (death, hospitalization, life-threatening, etc.), and patient demographics including age distribution and sex breakdown.

**Tab 2 - Trend Forecast**
Shows quarterly reporting volume going back to 2020, combining local database data with live OpenFDA API queries. Both a linear regression and a decision tree model are shown on the chart so you can see how they compare on historical data. Linear regression is used for the two-quarter forecast since decision trees cannot predict beyond their training range. Evaluation metrics and forecast cards for both models are shown below the chart.

**Tab 3 - Reaction Clusters**
Uses K-Means clustering on TF-IDF vectors to group reaction terms. PCA brings the high-dimensional vectors down to 2D so you can see the clusters on a scatter plot. Each point is one reaction term colored by body system, so you can check whether the clusters make clinical sense.

---

## Important Notes

- The severity and demographics charts need the local database to be populated. If Docker is not running or the ETL has not been run yet, those charts will show placeholder data and a warning banner will appear at the top of the dashboard.
- The reactions bar chart, trend forecast, and clustering tab always use the live OpenFDA API and will work regardless of database state.
- Results on the reactions tab may vary slightly between searches since the live API returns a sample of records rather than a fixed dataset.

---

## File Structure

```
FDA-Adverse-tracker/
├── app/
│   └── main.py              # Streamlit dashboard
├── data/
│   ├── raw/ASCII/           # downloaded FAERS quarterly files
│   ├── sample_drugs.csv     # sample data for quick setup
│   ├── sample_reports.csv
│   ├── sample_reactions.csv
│   └── sample_outcomes.csv
├── scripts/
│   ├── data_acquisition.py  # scrapes and downloads FAERS data from fda.gov
│   ├── db_config.py         # database connections and schema setup
│   ├── etl_processor.py     # ETL pipeline for loading FAERS CSVs into PostgreSQL
│   ├── models.py            # regression, clustering, and database query functions
│   └── openfda_api.py       # live OpenFDA API queries and MongoDB storage
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Database Credentials

Defaults match what is in docker-compose.yml. To override, set these environment variables:

| Variable | Default |
|---|---|
| POSTGRES_USER | faers_user |
| POSTGRES_PASSWORD | faers_password |
| POSTGRES_DB | faers_db |
| MONGO_USER | mongo_admin |
| MONGO_PASSWORD | mongo_password |

---

## Adminer (Database UI)

Once Docker is running, open http://localhost:8080 and log in with:
- System: PostgreSQL
- Server: postgres
- Username: faers_user
- Password: faers_password
- Database: faers_db

---

## Data Sources

- FAERS bulk download: https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html
- OpenFDA API: https://open.fda.gov/apis/drug/event/
```