import kagglehub

# Download latest version
path = kagglehub.dataset_download("greatgamedota/faceforensics")

print("Path to dataset files:", path)