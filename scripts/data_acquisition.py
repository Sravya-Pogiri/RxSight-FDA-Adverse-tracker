import os
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import zipfile

FDA_URL = 'https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')

def setup_directories():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

def find_latest_zip_links(url, limit=4):
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"couldn't reach FDA site: {e}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    links = soup.find_all('a', href=True)
    
    zip_links = []
    for link in links:
        href = link['href']
        if href.endswith('.zip') and 'ascii' in href.lower() and ('2024' in href or '2025' in href):
            if href not in zip_links:
                zip_links.append(href)
                
    return zip_links[:limit]

def download_file(url, save_dir):
    local_filename = os.path.join(save_dir, url.split('/')[-1])
    
    try:
        with requests.get(url, stream=True, timeout=20) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(local_filename, 'wb') as f, tqdm(
                desc=local_filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        size = f.write(chunk)
                        bar.update(size)
        return local_filename
    except Exception as e:
        print(f"download failed for {url}: {e}")
        if os.path.exists(local_filename):
            os.remove(local_filename)
        return None

def unzip_file(zip_path, extract_dir):
    if not zip_path or not os.path.exists(zip_path):
        return
        
    print(f"unzipping {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        os.remove(zip_path)
    except zipfile.BadZipFile as e:
        print(f"bad zip: {zip_path} ({e})")

def run_pipeline():
    setup_directories()
    print("looking for FAERS download links...")
    links = find_latest_zip_links(FDA_URL)
    
    if not links:
        print("no matching ZIP links found")
        return
        
    print(f"found {len(links)} quarterly files")
    
    for link in links:
        if not link.startswith('http'):
            link = "https://fis.fda.gov" + link if link.startswith('/') else "https://fis.fda.gov/" + link
            
        print(f"\n{link}")
        file_path = download_file(link, RAW_DATA_DIR)
        
        if file_path:
            unzip_file(file_path, RAW_DATA_DIR)

if __name__ == "__main__":
    run_pipeline()

