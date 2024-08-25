# Use an official Linux base image
FROM debian:bookworm

# Set the working directory
WORKDIR /app

# Update packages and install necessary dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-venv python3-pip locales 

# Create a virtual environment
RUN python3 -m venv venv

# Activate the virtual environment
ENV PATH="/app/venv/bin:$PATH"

# Copy the requirements file (if you have one)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set the entry point command
CMD python3 cctv_analytics/main.py