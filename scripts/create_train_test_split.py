import os
import shutil
from sklearn.model_selection import train_test_split
import random

def create_train_test_split(source_dir, output_dir, test_size=0.2, random_state=42):
    """Split data into train and test sets"""
    
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    
    for split_dir in [train_dir, test_dir]:
        for class_name in ['real', 'fake']:
            os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)
    
    # Process each class
    for class_name in ['real', 'fake']:
        class_path = os.path.join(source_dir, class_name)
        if not os.path.exists(class_path):
            continue
            
        # Get all files
        files = [f for f in os.listdir(class_path) if f.endswith('.jpg')]
        
        # Split files
        train_files, test_files = train_test_split(
            files, test_size=test_size, random_state=random_state
        )
        
        print(f"{class_name}: {len(train_files)} train, {len(test_files)} test")
        
        # Copy files to respective directories
        for file in train_files:
            shutil.copy2(
                os.path.join(class_path, file),
                os.path.join(train_dir, class_name, file)
            )
            
        for file in test_files:
            shutil.copy2(
                os.path.join(class_path, file),
                os.path.join(test_dir, class_name, file)
            )

if __name__ == "__main__":
    create_train_test_split(
        source_dir="../data/Celeb-DF/processed_frames",
        output_dir="../data/Celeb-DF/split_data",
        test_size=0.2
    )
