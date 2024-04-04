import os
import re
import csv
from datetime import datetime

def parse_datetime(timestamp):

    formats = ["%Y-%m-%d %H:%M:%S,%f", "%Y-%m-%d %H:%M:%S"]  # With and without milliseconds
    for fmt in formats:
        try:
            return datetime.strptime(timestamp, fmt)
        except ValueError:
            continue
    return None  # If neither format works

def calculate_time_taken(start, end):
  
    if not start or not end:
        return "N/A"
    delta = end - start
    total_seconds = delta.total_seconds()
    minutes = total_seconds / 60  # Convert seconds to minutes
    return f"{minutes:.2f}"

def parse_log_file(log_file_path):
    with open(log_file_path, 'r') as file:
        log_content = file.read()

        # Using regular expressions to extract required information
        efficientnet_variant = re.search(r"efficientnet_variant': '(\w+)'", log_content)
        batch_size = re.search(r"'batch_size': (\d+)", log_content)
        reg_lambda = re.search(r"'l2_lambda': (0.\d+)", log_content)
        learning_rate = re.search(r"'learning_rate': ([\d.]+)", log_content)
        optimizer = re.search(r"'optimizer': '(\w+)'", log_content)
        data_size = re.search(r"Loaded (\d+) images and", log_content)
        data_size_augmented = re.search(r"Size after augmentation: (\d+)", log_content)
        resize = re.search(r"Images resized to (\d+x\d+)", log_content)
        train_loss = re.search(r"Final Train loss: ([\d.]+)", log_content)
        val_loss = re.search(r"Final Val loss: ([\d.]+)", log_content)
        train_accuracy = re.search(r"Final Train accuracy: ([\d.]+)", log_content)
        val_accuracy = re.search(r"Final Val accuracy: ([\d.]+)", log_content)
        datetime_format = "%Y-%m-%d %H:%M:%S,%f"
        model_loaded_match = re.search(r"(\d+-\d+-\d+ \d+:\d+:\d+(?:,\d+)?) - INFO - Model loaded and base layers frozen", log_content)
        training_complete_match = re.search(r"(\d+-\d+-\d+ \d+:\d+:\d+(?:,\d+)?) - INFO - Training complete", log_content)
        print(model_loaded_match,UnicodeTranslateError)

        # Parse the datetime objects
        start_time = parse_datetime(model_loaded_match.group(1)) if model_loaded_match else None
        end_time = parse_datetime(training_complete_match.group(1)) if training_complete_match else None
        print(start_time)
        print(end_time)
        # Calculate the time taken
        time_taken = calculate_time_taken(start_time, end_time)
        log_file = os.path.basename(log_file_path)
        return [
            log_file,
            efficientnet_variant.group(1) if efficientnet_variant else "N/A",
            batch_size.group(1) if batch_size else "N/A",
            reg_lambda.group(1) if reg_lambda else "N/A",
            optimizer.group(1) if optimizer else "N/A",
            learning_rate.group(1) if learning_rate else "N/A",
            data_size.group(1) if data_size else "N/A",
            data_size_augmented.group(1) if data_size_augmented else "N/A",
            resize.group(1) if resize else "N/A",
            train_loss.group(1) if train_loss else "N/A",
            val_loss.group(1) if val_loss else "N/A",
            train_accuracy.group(1) if train_accuracy else "N/A",
            val_accuracy.group(1) if val_accuracy else "N/A",
            time_taken,
        ]

# Specify the directory containing the log files
log_dir = "logs/hyperparameter_tuning-b4-success"
# Update the log_files list comprehension to include a regex match for the expected file name format
log_files = [
    os.path.join(log_dir, f) for f in os.listdir(log_dir)
    if os.path.isfile(os.path.join(log_dir, f)) and re.match(r'Hyperparameter-Result', f)
]
print(log_files)
# CSV file to write the data
csv_file = "hyperparameter_results-b4.csv"

# Write header to CSV
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Log_file", "Efficientnet", "Batch Size", "Reg. Lambda", "Optimizer", "Learning Rate","Data Size", "Data Size after augmentation", "Resize", "Train Loss", "Test Loss", "Train Accuracy", "Test Accuracy", "Time Taken"])

    # Parse each log file and write the data to the CSV
    for log_file in log_files:
        try:
            row = parse_log_file(log_file)
            writer.writerow(row)
        except Exception as e:
            print(f"Error processing file {log_file}: {e}")

print(f"CSV file {csv_file} has been created with the hyperparameter tuning results.")
