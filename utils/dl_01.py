import requests
import os
from tqdm import tqdm

# The Collection ID for "NONAN GaitPrint"
COLLECTION_ID = [6415061]
OUTPUT_DIR = os.path.join("DATA","ya")


def download_collection(c_id):
    # Create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Directory '{OUTPUT_DIR}' ready.\n")

    print("Fetching file list from Figshare API...")
    
    # 1. Get the list of articles
    api_url = f"https://api.figshare.com/v2/collections/{c_id}/articles?page_size=100"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        articles = response.json()
    except Exception as e:
        print(f"Error fetching collection info: {e}")
        return

    # 2. Setup the Overall Progress Bar (Outer Loop)
    # position=0 keeps this bar at the top.
    with tqdm(total=len(articles), desc="Overall Progress", unit="file", position=0) as overall_bar:
        
        for article in articles:
            try:
                # Fetch details for the specific article to get the download URL
                details_url = article['url']
                article_details = requests.get(details_url).json()

                for file_info in article_details.get('files', []):
                    download_url = file_info['download_url']
                    file_name = file_info['name']
                    save_path = os.path.join(OUTPUT_DIR, file_name)

                    # Check if file already exists to skip re-downloading
                    if os.path.exists(save_path):
                        # Update the description to show we skipped a file, but don't disrupt the bar
                        overall_bar.write(f"Skipping {file_name} (already exists)")
                        continue
                    
                    # 3. Setup the Single File Progress Bar (Inner Loop)
                    # We send a HEAD request first or just stream to get the content-length
                    response = requests.get(download_url, stream=True)
                    total_size_in_bytes = int(response.headers.get('content-length', 0))
                    
                    # position=1 puts this bar below the overall one.
                    # leave=False makes it disappear after the file is done so the terminal stays clean.
                    with tqdm(
                        total=total_size_in_bytes, 
                        unit='iB', 
                        unit_scale=True, 
                        desc=file_name, 
                        position=1, 
                        leave=False
                    ) as file_bar:
                        
                        with open(save_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    file_bar.update(len(chunk))
            
            except Exception as e:
                overall_bar.write(f"Error downloading article {article.get('id', 'unknown')}: {e}")
            
            # Advance the overall progress bar by 1 article
            overall_bar.update(1)

    print("\n\nAll downloads complete!")

if __name__ == "__main__":
    for _ in COLLECTION_ID:
        download_collection(_)