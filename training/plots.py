import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import logging 
from datetime import datetime
from sklearn.metrics import confusion_matrix, roc_curve


def plot_loss(history,output_dir,title):
  pth = os.path.join(output_dir,"loss")
  os.makedirs(pth, exist_ok=True)
  now = datetime.now()
  filename = f"Loss_{title}_{now.strftime('%m-%d_%H-%M')}.png"
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  plt.title(title)
  plt.savefig(os.path.join(pth,filename))
  logging.info(f"Final Train loss: {history.history['loss'][-1]}")
  logging.info(f"Final Val loss: {history.history['val_loss'][-1]}")
  plt.close()

def plot_accuracy(history,output_dir,title):
  pth = os.path.join(output_dir,"accuracy")
  os.makedirs(pth, exist_ok=True)
  now = datetime.now()
  filename = f"Accuracy_{title}_{now.strftime('%m-%d_%H-%M')}.png"
  plt.plot(history.history['accuracy'], label='accuracy')
  plt.plot(history.history['val_accuracy'], label='val_accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.title(title)
  plt.savefig(os.path.join(pth,filename))
  logging.info(f"Final Train accuracy: {history.history['accuracy'][-1]}")
  logging.info(f"Final Val accuracy: {history.history['val_accuracy'][-1]}")
  plt.close()
  
  
def plot_confusion_matrix(y_true, y_pred, output_dir, title):
  pth = os.path.join(output_dir,"Confusion Matrix")
  os.makedirs(pth, exist_ok=True)
  now = datetime.now()
  filename = f"Confusion_matrix_{title}_{now.strftime('%m-%d_%H-%M')}.png"
  cm = confusion_matrix(y_true, y_pred)
  sns.heatmap(cm, annot=True, fmt='d')
  plt.xlabel('Predicted')
  plt.ylabel('Truth')
  plt.title(title)
  plt.savefig(os.path.join(pth,filename))
  plt.close()
  logging.info(f"Confusion matrix saved to {os.path.join(output_dir,filename)}")
  
def plot_roc_curve(y_true, y_pred, output_dir, title):
  pth = os.path.join(output_dir,"ROC")
  os.makedirs(pth, exist_ok=True)
  now = datetime.now()
  filename = f"ROC_{title}_{now.strftime('%m-%d_%H-%M')}.png"
  fpr, tpr, thresholds = roc_curve(y_true, y_pred)
  plt.plot(fpr, tpr)
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title(title)
  plt.savefig(os.path.join(pth,filename))
  plt.close()
  logging.info(f"ROC curve saved to {os.path.join(output_dir,filename)}")
