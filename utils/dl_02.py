import requests
import os
from tqdm import tqdm

# --- CONFIGURATION ---
ARTICLE_ID = 29371796 
OUTPUT_DIR = "NONAN_MiddleAged_Data"
# ---------------------

def download_article_files():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Directory '{OUTPUT_DIR}' checked.\n")

    print(f"Connecting to Figshare Article {ARTICLE_ID}...")

    # FIX: Added '?page_size=100' to get all files (default is only 10)
    api_url = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}/files?page_size=100"

    try:
        response = requests.get(api_url)
        response.raise_for_status()
        files_list = response.json()
    except Exception as e:
        print(f"Error fetching article info: {e}")
        return

    print(f"Found {len(files_list)} files. Starting download...")

    # Overall Progress Bar
    with tqdm(total=len(files_list), desc="Total Progress", unit="file", position=0) as pbar_overall:
        
        for file_info in files_list:
            file_name = file_info['name']
            download_url = file_info['download_url']
            save_path = os.path.join(OUTPUT_DIR, file_name)
            
            try:
                # Get file size first
                r = requests.get(download_url, stream=True)
                total_size = int(r.headers.get('content-length', 0))

                # Skip if already exists
                if os.path.exists(save_path):
                    if os.path.getsize(save_path) == total_size:
                        pbar_overall.write(f"Skipping {file_name} (already exists)")
                        pbar_overall.update(1)
                        r.close()
                        continue

                # Download with progress bar
                with tqdm(
                    total=total_size, 
                    unit='iB', 
                    unit_scale=True, 
                    desc=file_name, 
                    position=1, 
                    leave=False
                ) as pbar_file:
                    with open(save_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar_file.update(len(chunk))
            
            except Exception as e:
                pbar_overall.write(f"Failed to download {file_name}: {e}")

            pbar_overall.update(1)

    print("\nAll downloads complete!")

if __name__ == "__main__":
    download_article_files()