# Use an official Python runtime as a parent image
FROM python:3.12.3-slim

# Install required system packages for OpenCV
RUN apt-get update && apt-get install -y libgl1

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install dependencies globally
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 8080

# Run the app
CMD ["python", "app.py"]
