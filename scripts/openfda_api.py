import requests
from .db_config import get_mongo_client

OPEN_FDA_API_URL = "https://api.fda.gov/drug/event.json"

def fetch_drug_events(drug_name, limit=100, api_key=None):
    params = {
        "search": f'patient.drug.medicinalproduct:"{drug_name}"',
        "limit": limit
    }
    # api key is optional — unauthenticated requests work but hit rate limits faster
    if api_key:
        params["api_key"] = api_key

    print(f"querying openFDA for {drug_name} (limit={limit})...")
    response = requests.get(OPEN_FDA_API_URL, params=params)
    response.raise_for_status()

    results = response.json().get("results", [])
    print(f"got {len(results)} records back")
    return results

def store_raw_events_to_mongo(drug_name, events):
    if not events:
        return

    client = get_mongo_client()
    collection = client.faers_db.raw_api_events

    # tag each event with the drug name so we know what query produced it
    for event in events:
        event["queried_drug"] = drug_name

    collection.insert_many(events)
    print(f"wrote {len(events)} events to mongo")

def run_api_ingestion(drug_name, limit=100, api_key=None):
    events = fetch_drug_events(drug_name, limit, api_key)
    store_raw_events_to_mongo(drug_name, events)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_api_ingestion(sys.argv[1])
    else:
        print("Usage: python -m scripts.openfda_api <drug_name>")