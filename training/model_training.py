from sklearn.model_selection import train_test_split
import tensorflow as tf
import json
import os
import logging
from pathlib import Path
import sys
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from logs.enablelogging import setup_logging, close_logging
from load_dataset import load_data, n_partition_data,  split_data
from plots import plot_loss, plot_accuracy, plot_confusion_matrix, plot_roc_curve


def load_config(config_path):
    try:
        with open(config_path) as config_file:
            return json.load(config_file)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON config file: {e}")
        exit(1)


def compile_model(model, config):
    try:
        optimizer = config['model']['optimizer']
        loss = config['model']['loss']
        metrics = config['model']['metrics']
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model
    except KeyError as e:
        logging.error(f"Missing key in config: {e}")
        raise

def enable_gpu(memory=14):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Set GPU memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            # Set GPU memory limit to 14 GB
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory*1024)]
            )
            logging.info("GPU enabled of memory GB : ",memory)
            tf.config.run_functions_eagerly(True)
        except RuntimeError as e:
            logging.error(f"Error enabling GPU: {e}")

def create_callbacks(config):
    lr = config['model']['learning_rate']
    decay = config['model']['decay']
    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(lambda epoch: lr * decay ** epoch),
        tf.keras.callbacks.ModelCheckpoint(filepath='model_checkpoint.h5', save_best_only=True)
    ]
    return callbacks

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


def train_model(model, X_train, y_train, X_val, y_val, config):
    epochs = config['model']['epochs']
    batch_size = config['model']['batch_size']
    callbacks = create_callbacks(config)

    try:
        history = model.fit(X_train, y_train,
                            validation_data=(X_val, y_val),
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=callbacks)
        return history,model
    except Exception as e:  # Catch generic exceptions for training issues
        logging.error(f"Error during training: {e}")
        raise


def save_model(model, config):
    model_output_path = config['model']['output_path']
    try:
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        model.save(model_output_path)
    except Exception as e:  # Catch generic exceptions for saving issues
        logging.error(f"Error saving model: {e}")


def plot_results(history, output_dir, plot_title, plot_types,new_folder_flag, plot_dict):
    if new_folder_flag:
        now = datetime.now()
        output_dir = os.path.join(output_dir, now.strftime('%m-%d_%H-%M'))
    for plot_type in plot_types:
        if plot_type in plot_dict:
            try:
                plot_dict[plot_type](history, output_dir, plot_title)
                logging.info(f"{plot_type} plot saved to {output_dir}")
            except Exception as e:  # Catch generic exceptions for plotting issues
                logging.warning(f"Error generating plot {plot_type}: {e}")
        else:
            logging.warning(f"Unsupported plot type: {plot_type}")


def main(config_path, log_dir="logs/model_training"):
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
        setup_logging(log_dir)
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
        fake_limit = config['fake_limit']
        data_augmentation = config['data_augmentation']['flag']
        data_augmentation_ratio = config['data_augmentation']['ratio']
        partition_flag = config['partition']['flag']
        
        gpu_mem = config.get("gpu_mem", 14)
        
        if partition_flag:
            n = config['partition']['n']
            p = config['partition']['p']
            real_limit_scale = config['partition']['real_limit_scale']
            fake_to_real_ratio = config['partition']['fake_to_real_ratio']
            format = config['partition']['format']
            if data_augmentation:
                X, y = n_partition_data(real_dir, fake_dir, n, p, real_limit_scale, fake_to_real_ratio, format, data_augmentation, data_augmentation_ratio,resize=resize)
            else:
                X, y = n_partition_data(real_dir, fake_dir, n, p, real_limit_scale, fake_to_real_ratio, format, resize=resize)
            logging.info(f"Partitioned data into {n} parts, using part {p}")
        else:
            if data_augmentation:
                X, y = load_data(real_dir, fake_dir, real_limit, fake_limit, data_augmentation, resize=resize)
            else:   
                X, y = load_data(real_dir, fake_dir ,real_limit,fake_limit,resize=resize)
        
        val_split = config['val_split']
        X_train, X_val, y_train, y_val = split_data(X, y, val_split)
        logging.info(f"Split data into training and validation sets with {len(X_train)} and {len(X_val)} samples respectively")
        model = load_efficeintnet_model(efficientnet_variant)
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
    main(r"config\model_training\train_config.json")
