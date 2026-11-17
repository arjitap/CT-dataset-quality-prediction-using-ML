import os
import shutil
import numpy as np

def populate_folders_from_splits():
    # 1. Define paths
    splits_dir = "data/splits"
    train_out_dir = "training data"
    test_out_dir = "testing data"
    
    os.makedirs(train_out_dir, exist_ok=True)
    os.makedirs(test_out_dir, exist_ok=True)
    
    print("Checking split ID files inside 'data/splits/'...")
    
    # Check for typical split ID filenames (adjust if your filenames differ slightly, e.g., 'train_id.txt' or similar)
    train_id_files = [f for f in os.listdir(splits_dir) if "train" in f.lower()] if os.path.exists(splits_dir) else []
    test_id_files = [f for f in os.listdir(splits_dir) if "test" in f.lower()] if os.path.exists(splits_dir) else []
    
    print(f"Found Train split references: {train_id_files}")
    print(f"Found Test split references: {test_id_files}")
    
    # 2. Look for the raw dataset files or features that correspond to these IDs
    # (Assuming your base data is stored in 'data/' or similar directory)
    base_data_dir = "data"
    
    # Helper to load ID lists (whether text files, csv, or numpy arrays)
    def load_ids(file_path):
        if file_path.endswith('.txt') or file_path.endswith('.csv'):
            with open(file_path, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        elif file_path.endswith('.npy') or file_path.endswith('.npz'):
            return list(np.load(file_path, allow_pickle=True))
        return []

    # Process Training IDs
    train_copied_count = 0
    for tf in train_id_files:
        tf_path = os.path.join(splits_dir, tf)
        ids = load_ids(tf_path)
        print(f"Loaded {len(ids)} IDs from {tf}")
        
        for item_id in ids:
            # Find matching file in base data or source folders
            # Adjust pattern depending on how your scan files are named (e.g., scan_{item_id}.npz)
            potential_matches = [
                os.path.join(base_data_dir, f"{item_id}.npz"),
                os.path.join(base_data_dir, f"scan_{item_id}.npz"),
                os.path.join(base_data_dir, item_id)
            ]
            for match in potential_matches:
                if os.path.exists(match):
                    shutil.copy(match, os.path.join(train_out_dir, os.path.basename(match)))
                    train_copied_count += 1
                    break

    # Process Testing IDs
    test_copied_count = 0
    for tef in test_id_files:
        tef_path = os.path.join(splits_dir, tef)
        ids = load_ids(tef_path)
        print(f"Loaded {len(ids)} IDs from {tef}")
        
        for item_id in ids:
            potential_matches = [
                os.path.join(base_data_dir, f"{item_id}.npz"),
                os.path.join(base_data_dir, f"scan_{item_id}.npz"),
                os.path.join(base_data_dir, item_id)
            ]
            for match in potential_matches:
                if os.path.exists(match):
                    shutil.copy(match, os.path.join(test_out_dir, os.path.basename(match)))
                    test_copied_count += 1
                    break

    print(f"\n--- Summary ---")
    print(f"Successfully populated '{train_out_dir}/' with {train_copied_count} scan files.")
    print(f"Successfully populated '{test_out_dir}/' with {test_copied_count} scan files.")

if __name__ == "__main__":
    populate_folders_from_splits()