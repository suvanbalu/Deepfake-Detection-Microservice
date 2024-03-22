import subprocess
import json
import tempfile
import os

# Define hyperparameters to test
learning_rates = [0.001, 0.0001,0.1,0.01]
batch_sizes = [12,16,32]
regularizations = [
    {"flag": True, "type": "l2", "lambda": 0.001},
    {"flag": True, "type": "l2", "lambda": 0.01},
    {"flag": True, "type": "l2", "lambda": 0.1},
    {"flag": True, "type": "l2", "lambda": 0},
]
optimizers = ["adam", "sgd"]
efficientnet_variants = ["b0", "b1", "b2", "b3", "b4", "b5","b6","b7"]

# Generate all combinations of hyperparameters
from itertools import product
hyperparameter_combinations = list(product(learning_rates, batch_sizes, regularizations, optimizers,efficientnet_variants))

for lr, batch_size, reg, optimizer,effnet_variant in hyperparameter_combinations:
    # Update base configuration with the new set of hyperparameters
    config = {
        "efficientnet_variant": effnet_variant,
        "data_dir": "combined_faces",
        "data_augmentation": {"flag": True, "ratio": 0.5},
        "real_limit": 2500,
        "fake_limit_scale": 1.5,
        "partition": {"flag": False, "n": 1, "p": 1, "real_limit_scale": -1, "fake_to_real_ratio": 1, "format": "jpg"},
        "gpu_mem": 14,
        "val_split": 0.2,
        "model": {
            "epochs": 30,
            "batch_size": batch_size,
            "learning_rate": lr,
            "decay": 0.9,
            "optimizer": optimizer,
            "loss": "binary_crossentropy",
            "metrics": ["accuracy"],
            "output_path": "./faces_trained_model.h5",
            "regularization": {
                "flag": reg["flag"],
                "type": reg["type"],
                "l2_lambda": reg["lambda"] if reg["type"] == "l2" else None,
                "l1_lambda": reg["lambda"] if reg["type"] == "l1" else None,
            }
        },
        "plot": {
            "flag": True,
            "output_dir": "plots",
            "new_folder_flag": True,
            "types": ["loss", "accuracy"],
            "title": f"LR={lr}_BS={batch_size}_{reg['type'].upper()}={reg['lambda']}_OPT={optimizer}_Effnet={effnet_variant}"
        }
    }

    # Create a temporary file to save the updated config
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmpfile:
        json.dump(config, tmpfile)
        tmpfile_path = tmpfile.name

    # Run the training script with the updated config
    try:
        subprocess.run(['python', 'model_training.py', tmpfile_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with config: LR={lr}, Batch Size={batch_size}, Reg={reg['type']}={reg['lambda']}, Optimizer={optimizer}")
        print(e)
    finally:
        # Clean up the temporary file
        os.remove(tmpfile_path)