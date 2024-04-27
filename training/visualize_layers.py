import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
import sys
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import os

def load_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

def build_model():
    inputs = tf.keras.Input(shape=(224, 224, 3))
    # Load the base model, without the top layers
    base_model = tf.keras.applications.EfficientNetB0(include_top=False, input_tensor=inputs, weights='imagenet')
    base_model.trainable = False

    # Add new layers on top
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    print(model.summary())
    return model

def build_intermediate_model(model):
    # Creates a new model that outputs the intermediate layers' outputs
    layer_outputs = [layer.output for layer in model.layers]  # Get all layer outputs
    intermediate_model = Model(inputs=model.input, outputs=layer_outputs)
    return intermediate_model

def visualize_layer_outputs(intermediate_model, input_data, output_dir='layer_outputs'):
    intermediate_outputs = intermediate_model.predict(input_data)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, (output, layer) in enumerate(zip(intermediate_outputs, intermediate_model.layers)):
        layer_name = layer.name  # Or use `type(layer).__name__` for type like 'Conv2D', 'Dense', etc.
        if len(output.shape) == 4:
            # This is a feature map from a convolutional layer
            num_channels = output.shape[-1]
            for channel in range(num_channels):
                channel_output = output[0, :, :, channel]
                plt.imshow(channel_output, cmap='viridis')
                # Using layer name in the directory path and file name
                channel_output_dir = os.path.join(output_dir, f"layer_{i}_{layer_name}")
                os.makedirs(channel_output_dir, exist_ok=True)
                plt.savefig(f"{channel_output_dir}/channel_{channel}.png")
                plt.close()
        else:
            # For layers outputting 2D data or less, like Dense layers
            plt.plot(output[0, :], marker='o')
            plt.title(f"Layer {i} ({layer_name}) Output")
            # Include layer name in the output file name
            plt.savefig(f"{output_dir}/layer_{i}_{layer_name}_output.png")
            plt.close()
            
def visualize_selected_layers(intermediate_model, input_data, base_model_len, output_dir='selected_layer_outputs'):
    intermediate_outputs = intermediate_model.predict(input_data)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Iterate over outputs, skipping to layers after the base model
    for i, output in enumerate(intermediate_outputs):
        if i>0 and i<base_model_len-1:
            continue
        layer_name = intermediate_model.layers[i].name
        if len(output.shape) == 4:  # Convolutional layers
            for channel in range(min(output.shape[-1], 10)):  # Visualize up to 10 channels
                channel_output = output[0, :, :, channel]
                plt.imshow(channel_output, cmap='viridis')
                layer_output_dir = os.path.join(output_dir, f"layer_{i}_{layer_name}")
                os.makedirs(layer_output_dir, exist_ok=True)
                plt.title(f"{layer_name} - Channel {channel}")
                plt.savefig(f"{layer_output_dir}/channel_{channel}.png")
                plt.close()
        else:  # Dense layers or any layer with 2D output or less
            plt.plot(output[0, :], marker='o')
            plt.title(f"{layer_name} Output")
            plt.savefig(f"{output_dir}/layer_{i}_{layer_name}_output.png")
            plt.close()

if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) == 2 else "test/fake4_mtcnn.png"
    input_data = load_preprocess_image(img_path)
    model = build_model()
    intermediate_model = build_intermediate_model(model)
    # visualize_layer_outputs(intermediate_model, input_data)
    base_model = tf.keras.applications.EfficientNetB0(include_top=False, input_tensor=tf.keras.Input(shape=(224, 224, 3)), weights='imagenet')
    len_base = len(base_model.layers)
    visualize_selected_layers(intermediate_model,input_data,len_base)
