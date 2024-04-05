import tensorflow as tf
import json
import os
import logging
from pathlib import Path
import sys
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from logs.enablelogging import setup_logging, close_logging
from load_dataset import load_data, n_partition_data, split_data
from plots import plot_loss, plot_accuracy, plot_confusion_matrix, plot_roc_curve
from tensorflow.keras import layers
from tensorflow.keras import regularizers

from utils import load_config,compile_model,enable_gpu, train_model,create_callbacks, save_model, plot_results

def load_efficeintnet_model(efficientnet_variant):
  if efficientnet_variant == "b0":
    model = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet')
  elif efficientnet_variant == "b1":
    model = tf.keras.applications.EfficientNetB1(include_top=False, weights='imagenet')
  elif efficientnet_variant == "b2":
    model = tf.keras.applications.EfficientNetB2(include_top=False, weights='imagenet')
  elif efficientnet_variant == "b3":
    model = tf.keras.applications.EfficientNetB3(include_top=False, weights='imagenet')
  elif efficientnet_variant == "b4":
    model = tf.keras.applications.EfficientNetB4(include_top=False, weights='imagenet')
  elif efficientnet_variant == "b5":
    model = tf.keras.applications.EfficientNetB5(include_top=False, weights='imagenet')
  elif efficientnet_variant == "b6":
    model = tf.keras.applications.EfficientNetB6(include_top=False, weights='imagenet')
  elif efficientnet_variant == "b7":
    model = tf.keras.applications.EfficientNetB7(include_top=False, weights='imagenet')
  else:
    logging.error(f"Unsupported EfficientNet variant: {efficientnet_variant}")
    raise ValueError(f"Unsupported EfficientNet variant: {efficientnet_variant}")
  model.trainable = False
  return model

def get_regularizer(config):
    if config['model']['regularization']['flag']:
        reg_type = config['model']['regularization']['type']
        if reg_type == 'l2':
            return regularizers.l2(config['model']['regularization']['l2_lambda'])
        elif reg_type == 'l1':
            return regularizers.l1(config['model']['regularization']['l2_lambda'])
        else:
            logging.warning(f"Unsupported regularization type: {reg_type}")
            return None
    else:
        return None
      
def build_model(base_model, config):
    regularizer = get_regularizer(config)
    model = tf.keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizer),  
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  
    ])
    return model     


def main(config_path, log_dir="logs/model_training",log_name=None):
  model_resize = {
      "b0": 224,
      "b1": 240,
      "b2": 260,
      "b3": 300,
      "b4": 320,
      "b5": 320,
      "b6": 400,
      "b7": 520
  }
  try:
    setup_logging(log_dir,log_name)
  except Exception as e:
    print("Error setting up logging: ", e)
    exit(1)
  try:
    config = load_config(config_path)
    logging.info(f"Using configuration: {config}")
    efficientnet_variant = config['efficientnet_variant']
    resize = model_resize[efficientnet_variant]
    data_dir = config['data_dir']
    real_dir = Path(data_dir) / 'REAL'
    fake_dir = Path(data_dir) / 'FAKE'
    real_limit = config['real_limit']
    fake_limit_scale = config['fake_limit_scale']
    data_augmentation = config['data_augmentation']['flag']
    data_augmentation_ratio = config['data_augmentation']['ratio']
    partition_flag = config['partition']['flag']

    gpu_mem = config.get("gpu_mem", 14)
    enable_gpu(gpu_mem)
    if partition_flag:
      n = config['partition']['n']
      p = config['partition']['p']
      real_limit_scale = config['partition']['real_limit_scale']
      fake_to_real_ratio = config['partition']['fake_to_real_ratio']
      format = config['partition']['format']
      if data_augmentation:
        X, y = n_partition_data(real_dir, fake_dir, n, p, real_limit_scale, fake_to_real_ratio, format, data_augmentation, data_augmentation_ratio, resize=resize)
      else:
        X, y = n_partition_data(real_dir, fake_dir, n, p, real_limit_scale, fake_to_real_ratio, format, resize=resize)
      logging.info(f"Partitioned data into {n} parts, using part {p}")
    else:
      if data_augmentation:
        X, y = load_data(real_dir, fake_dir, real_limit, fake_limit_scale, data_augmentation, resize=resize)
      else:
        X, y = load_data(real_dir, fake_dir, real_limit, fake_limit_scale, resize=resize)

    val_split = config['val_split']
    X_train, X_val, y_train, y_val = split_data(X, y, val_split)

    logging.info(f"Split data into training and validation sets with {len(X_train)} and {len(X_val)} samples respectively")

    base_model = load_efficeintnet_model(efficientnet_variant)
    model = build_model(base_model,config)
    logging.info("Model loaded and base layers frozen")

    # Compile the model with L1 or L2 regularization based on configuration
    model = compile_model(model, config)

    history, model = train_model(model, X_train, y_train, X_val, y_val, config)
    logging.info("Training complete")
    save_model(model, config)

    plot_flag = config["plot"]["flag"]
    if plot_flag:
      new_folder_flag = config["plot"].get("new_folder_flag", False)
      output_dir = config["plot"]["output_dir"]
      plot_types = config["plot"]["types"]
      plot_title = config["plot"]["title"]
      plot_results(history, output_dir, plot_title, plot_types, new_folder_flag, plot_dict={"loss": plot_loss, "accuracy": plot_accuracy})
      # Add other plot functions to the dictionary as needed
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph() 
    logging.info("Model training complete")
  except Exception as e:
    logging.error(f"An error occurred: {e}")
    print("An error occurred: ", e)
  finally:
    close_logging()


if __name__ == "__main__":
  
  if len(sys.argv) > 1:
    config_path = sys.argv[1]
    if len(sys.argv) > 2:
      log_dir = sys.argv[2]
      if len(sys.argv)>3:
        log_name=sys.argv[3]
  else:
    config_path = r"config/model_training/linux_train_config.json"
    log_dir = r"logs/final_training"
    log_name=None
  main(config_path, log_dir,log_name)