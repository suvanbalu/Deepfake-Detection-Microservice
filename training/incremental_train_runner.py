import json
import subprocess

data_directories = ["faces4", "faces2", "faces3"] 
log_dir = "logs/model_training/30epochs-b5" 
training_script = "training/incremental_training.py"  
for data_dir in data_directories:
  config = {
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

  # Save the configuration to a JSON file
  with open(f"config_{data_dir}.json", "w") as f:
      json.dump(config, f, indent=4)

  # Call your incremental training code here, passing the path to the config file as an argument
  subprocess.run(["python3", training_script, f"config_{data_dir}.json",log_dir])
  print(f"Created config file: config_{data_dir}.json")
  print(f"Executed training script with config_{data_dir}.json")

