import subprocess
import threading
import time

def train_model(script_name):
    """Train a single model"""
    print(f"Starting {script_name}...")
    result = subprocess.run(['python', f'models/{script_name}'], 
                          capture_output=True, text=True)
    print(f"Completed {script_name}")
    print(result.stdout)
    if result.stderr:
        print(f"Error in {script_name}: {result.stderr}")

# Model scripts to run
models = ['basic_cnn.py', 'efficientnet.py', 'resnet50.py', 'vgg16.py']

# Start all training in parallel
threads = []
start_time = time.time()

for model in models:
    thread = threading.Thread(target=train_model, args=(model,))
    thread.start()
    threads.append(thread)

# Wait for all to complete
for thread in threads:
    thread.join()

end_time = time.time()
print(f"\nAll models trained in {end_time - start_time:.2f} seconds")
