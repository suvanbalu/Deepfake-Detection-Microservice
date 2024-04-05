import os
import json
import logging
import tensorflow as tf
from datetime import datetime


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
      logging.info(f"GPU enabled of memory GB : {memory}")
      # tf.config.run_functions_eagerly(True)
    except RuntimeError as e:
      logging.error(f"Error enabling GPU: {e}")
    
def create_callbacks(config):
  lr = config['model']['learning_rate']
  decay = config['model']['decay']
  callbacks = [
      tf.keras.callbacks.LearningRateScheduler(lambda epoch: lr * decay ** epoch),
      tf.keras.callbacks.ModelCheckpoint(filepath='model_checkpoint.h5', save_best_only=True),
      tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True)
  ]
  return callbacks

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
    return history, model
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
    
def plot_results(history, output_dir, plot_title, plot_types, new_folder_flag, plot_dict):
  if new_folder_flag:
    now = datetime.now()
    output_dir = os.path.join(output_dir, now.strftime('%m-%d_%H-%M'))
  for plot_type in plot_types:
    if plot_type in plot_dict:
      try:
        plot_dict[plot_type](history, output_dir, plot_title)
        logging.info(f"{plot_type} plot saved to {output_dir}/{plot_type}")
      except Exception as e:  # Catch generic exceptions for plotting issues
        logging.warning(f"Error generating plot {plot_type}: {e}")
    else:
      logging.warning(f"Unsupported plot type: {plot_type}")
