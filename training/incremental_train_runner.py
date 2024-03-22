import json
import subprocess
import tempfile
import os

def create_config(data_dir):
    """Creates a configuration dictionary for the given data directory."""
    return {
        "model_path": "faces_trained_model.h5",
        "data_dir": data_dir,
        "data_augmentation": {
            "flag": True,
            "ratio": 0.5
        },
        "real_limit": -1,
        "fake_limit_scale": 1.5,
        "partition": {
            "flag": False,
            "n": 1,
            "p": 1,
            "real_limit_scale": -1,
            "fake_to_real_ratio": 1,
            "format": "jpg"
        },
        "val_split": 0.2,
        "model": {
            "epochs": 30,
            "batch_size": 32,
            "learning_rate": 0.01,
            "decay": 0.9,
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "metrics": ["accuracy"],
            "output_path": "./faces_trained_model.h5"
        },
        "plot": {
            "flag": True,
            "output_dir": "plots/30epochs-b5",
            "new_folder_flag": True,
            "types": ["loss", "accuracy"],
            "title": f"Incremental Training {data_dir} Results"
        }
    }

def run_training(training_script, config, log_dir):
    """Runs the training script with the given configuration file and log directory."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmpfile:
        json.dump(config, tmpfile)
        tmpfile_path = tmpfile.name
    
    # Execute the training script
    try:
        subprocess.run(["python3", training_script, tmpfile_path, log_dir], check=True)
        print(f"Executed training script with {tmpfile_path}")
    finally:
        # Optionally, you can choose to keep or remove the config file after training
        os.remove(tmpfile_path)

data_directories = ["faces49", "faces25", "faces35"] 
log_dir = "logs/model_training/30epochs-b5" 
training_script = "training/incremental_training.py"  

for data_dir in data_directories:
    config = create_config(data_dir)
    run_training(training_script, config, log_dir)