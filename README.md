# FDA Adverse Event Tracker

This is my CS project for tracking adverse drug events using FDA data. The idea is to pull data from the FDA's FAERS database, store it, and build a dashboard that lets you search for a drug and see what kinds of side effects get reported most often, who's reporting them, and whether reports are trending up or down over time.

---

## What this does

The project has a few main pieces:

1. Downloads quarterly FAERS data files from the FDA website and stores them locally
2. Runs an ETL script to clean and load everything into a PostgreSQL database
3. Also pulls live data from the OpenFDA API and stores raw results in MongoDB
4. Trains a simple linear regression model to forecast reporting trends
5. Uses K-Means clustering to group similar reactions together
6. Shows everything in a Streamlit dashboard with 3 tabs

---

## Setup

You'll need Python 3.9+, Docker, and pip.

**Step 1: Set up virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Step 2: Start the databases**

I used Docker to run Postgres and MongoDB locally so I didn't have to install them manually.

```bash
docker-compose up -d
```

This starts Postgres on port 5432, MongoDB on port 27017, and Adminer (a database UI) on port 8080.

**Step 3: Set up the database tables**

```bash
python -m scripts.db_config
```

**Step 4: Download the FAERS data**

```bash
python -m scripts.data_acquisition
```

This scrapes the FDA website and downloads the ZIP files for the most recent quarters, then unzips them into `data/raw/ASCII/`.

**Step 5: Run the ETL**

```bash
# runs in test mode (limits to 50k rows per file, way faster)
python -m scripts.etl_processor
```

**Step 6: Launch the dashboard**

```bash
streamlit run app/main.py
```

Then open `http://localhost:8501` in your browser.

---

## Dashboard

There are 3 tabs:

**Tab 1 - Search View:** Type in a drug name and you get a bar chart of the most common reported reactions, a breakdown of how severe those reports are (death, hospitalization, etc.), and charts showing the age range and sex of patients.

**Tab 2 - Trend View:** Shows a time-series of how many reports have been filed per quarter. The linear regression model is overlaid on the chart so you can see how well it fits. There's also a forecast for the next 2 quarters, plus some basic model stats (R², RMSE).

**Tab 3 - Cluster View:** Uses K-Means to group reactions into clusters based on how similar the terms are (using TF-IDF). Then uses PCA to compress the clusters down to 2D so they can be plotted as a scatter plot. Each point is a reaction term and it's colored by which body system it belongs to.

---

## File Structure

```
FDA-Adverse-tracker/
├── app/
│   └── main.py              # the streamlit app
├── data/
│   ├── raw/ASCII/           # where the downloaded FAERS files live
│   └── processed/
├── scripts/
│   ├── data_acquisition.py  # scrapes and downloads data from fda.gov
│   ├── db_config.py         # database connections and table setup
│   ├── etl_processor.py     # loads FAERS CSVs into postgres
│   ├── models.py            # regression, clustering, and DB query functions
│   └── openfda_api.py       # hits the live openFDA API and stores to mongo
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Database credentials

The defaults match what's in docker-compose.yml. If you want to change them, set environment variables:

- `POSTGRES_USER` (default: faers_user)
- `POSTGRES_PASSWORD` (default: faers_password)
- `POSTGRES_DB` (default: faers_db)
- `MONGO_USER` (default: mongo_admin)
- `MONGO_PASSWORD` (default: mongo_password)

---

## Data sources

- FAERS bulk download: https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html
- OpenFDA API: https://open.fda.gov/apis/drug/event/
