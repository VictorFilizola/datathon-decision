# Use a stable, long-term support Python version like 3.11
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# (This is still good practice) Install essential build tools just in case
RUN apt-get update && apt-get install -y gcc build-essential && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first
COPY requirements.txt .

# Install all the Python libraries
# This will now be much faster as it will use pre-compiled wheels
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project's code into the container
COPY . .

# Expose the ports for the API (8000) and Frontend (8501)
EXPOSE 8000
EXPOSE 8501