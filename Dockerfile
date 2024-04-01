FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the app.py, models directory, and requirements.txt into the container at /app
COPY app.py /app/
COPY models /app/models
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV PORT=5000

# Run app.py when the container launches using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]