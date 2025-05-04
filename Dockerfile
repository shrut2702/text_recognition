# Use an official Python runtime as a parent image
FROM python:3.12.3-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install virtualenv and create a virtual environment
RUN pip install --no-cache-dir virtualenv && \
    python -m venv /app/vir_env

# Set environment variables to use virtual environment by default
ENV PATH="/app/vir_env/bin:$PATH"

# Install dependencies using the virtual environment
RUN pip install -r requirements.txt

# Expose the port the app runs on
EXPOSE 8080

# Run the app using the virtual environment's Python
CMD ["python", "app.py"]
