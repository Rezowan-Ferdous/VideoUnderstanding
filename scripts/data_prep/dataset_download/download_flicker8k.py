import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# --- Configuration ---
DATA_DIR = r"C:\Users\rezow\Rezowan\Projects\VideoUnderstanding\data\flickr8k"
# DATA_DIR = "data/flickr8k"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
CAPTIONS_FILE = os.path.join(DATA_DIR, "captions.csv")
# ---------------------

def main():
    """
    Downloads the Flickr8k dataset, saves the images to a directory,
    and creates a single captions.csv file.
    """
    print("Downloading Flickr8k dataset from Hugging Face...")
    # ---
    # UPDATED: Using a correct, available dataset path
    # This dataset has one item per image, with a list of captions
    # ---
    try:
        dataset = load_dataset("tsystems/flickr8k", split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please check your internet connection and if 'HuggingFaceM4/flickr8k' is still available.")
        return

    print(f"Dataset downloaded. Found {len(dataset)} images.")
    
    os.makedirs(IMAGE_DIR, exist_ok=True)
    
    captions_data = []
    
    print(f"Processing and saving images to {IMAGE_DIR}...")
    # ---
    # UPDATED: Loop to handle the new dataset structure
    # ---
    for item in tqdm(dataset):
        # The image_id is the filename (e.g., "1000268201_693b08cb0e")
        image_file = item['image_filename'] 
        image_path = os.path.join(IMAGE_DIR, image_file)
        
        try:
            # Save the image
            item['image'].save(image_path)
        except Exception as e:
            print(f"Warning: Could not save image {image_file}. Error: {e}")
            continue
        
        # 'captions' is a list of strings
        for caption_text in item['captions']:
            if caption_text: # Ensure caption is not empty
                # Add one row per caption
                captions_data.append({
                    "image": image_file,
                    "caption": caption_text.strip()
                })

    # Save captions CSV
    df = pd.DataFrame(captions_data)
    
    # Clean up just in case
    df = df.dropna()
    df = df[df['caption'].str.len() > 0]
    
    df.to_csv(CAPTIONS_FILE, index=False)
    
    print("\n--- Success! ---")
    print(f"Saved {len(dataset)} unique images to {IMAGE_DIR}")
    print(f"Saved {len(df)} total captions to {CAPTIONS_FILE}")
    print("\nRun this command to check:")
    print(f"ls -l {DATA_DIR}")

if __name__ == "__main__":
    main()
