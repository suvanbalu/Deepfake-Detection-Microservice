import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import os
import logging
from pathlib import Path
import sys
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from logs.enablelogging import setup_logging, close_logging
from load_dataset import load_data, n_partition_data,split_data
from plots import plot_loss, plot_accuracy, plot_confusion_matrix, plot_roc_curve

from utils import load_config,compile_model,enable_gpu, train_model,save_model, plot_results

def freeze_base_layers(model,num=0):
    for layer in model.layers[:-num]:
        layer.trainable = False
    return model

def main(config_path, log_dir="logs/incremental_training"):
    try:
        setup_logging(log_dir)
    except Exception as e:
        print("Error setting up logging: ", e)
        exit(1)
    try:
        config = load_config(config_path)
        logging.info(f"Using configuration: {config}")
        model_path = config['model_path']
        data_dir = config['data_dir']
        real_dir = Path(data_dir) / 'REAL'
        fake_dir = Path(data_dir) / 'FAKE'
        real_limit = config['real_limit']
        fake_limit_scale = config['fake_limit_scale']
        resize = config.get('resize', 224)
        gpu_mem = config.get("gpu_mem", 14)
        enable_gpu(gpu_mem)

        data_augmentation = config['data_augmentation']['flag']
        data_augmentation_ratio = config['data_augmentation']['ratio']
        partition_flag = config['partition']['flag']
        
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
                X, y = load_data(real_dir, fake_dir ,real_limit,fake_limit_scale, resize=resize)
        
        val_split = config['val_split']
        X_train, X_val, y_train, y_val = split_data(X, y, val_split)
        logging.info(f"Split data into training and validation sets with {len(X_train)} and {len(X_val)} samples respectively")
        model = load_model(model_path)
        model = freeze_base_layers(model,4)
        logging.info("Model loaded and base layers frozen")
        model = compile_model(model, config)
        history,model = train_model(model, X_train, y_train, X_val, y_val, config)
        logging.info("Training complete")
        save_model(model, config)

        plot_flag = config["plot"]["flag"]
        if plot_flag:
            new_folder_flag = config["plot"].get("new_folder_flag", False)
            output_dir = config["plot"]["output_dir"]
            plot_types = config["plot"]["types"]
            plot_title = config["plot"]["title"]
            plot_results(history, output_dir, plot_title, plot_types,new_folder_flag, plot_dict={"loss": plot_loss, "accuracy": plot_accuracy})
            # Add other plot functions to the dictionary as needed

        logging.info("Incremental training complete")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print("An error occurred: ", e)
    finally:
        close_logging()



if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        if len(sys.argv)>2:
            log_dir = sys.argv[2]
    else:
        config_path = "config/incremental_training/linux_inc_train.json"
        log_dir = "logs/incremental_training" 
    main(config_path,log_dir)