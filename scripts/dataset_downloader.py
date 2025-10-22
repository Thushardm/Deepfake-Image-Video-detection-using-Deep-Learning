import kagglehub
import os

def download_dataset(dataset_name):
    """Download specified dataset"""
    datasets = {
        'faceforensics': 'greatgamedota/faceforensics',
        'celebdf': 'reubensuju/celeb-df-v2', 
        'lfw': 'jessicali9530/lfw-dataset'
    }
    
    if dataset_name not in datasets:
        print(f"Available datasets: {list(datasets.keys())}")
        return None
        
    path = kagglehub.dataset_download(datasets[dataset_name])
    print(f"Downloaded {dataset_name} to: {path}")
    return path

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        download_dataset(sys.argv[1])
    else:
        print("Usage: python dataset_downloader.py <dataset_name>")
