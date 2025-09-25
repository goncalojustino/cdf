# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
# PYTHONUNBUFFERED: Prevents Python from buffering stdout and stderr
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (if any).
# For example, if your CDF processing libraries need system packages like
# build-essential or libgfortran-dev, you can uncomment the following lines:
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     libgfortran-dev \
#  && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create a directory for test data and copy the test file into it
RUN mkdir /app/test_data
COPY oldfiles/kh18.cdf /app/test_data/

# Copy the rest of the application's code
COPY . .

# Expose the port the app runs on
# Gunicorn will be configured to run on this port.
EXPOSE 8000

# Define the command to run the application
# This command starts gunicorn, a production-grade WSGI server.
# -w 4: Use 4 worker processes. Adjust as needed for your CPU cores.
# -b 0.0.0.0:8000: Bind to all network interfaces on port 8000.
# app:server: 'app' is your Python file (app.py), and 'server' is the
#             WSGI-compatible application object inside it (app.server).
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "app:server"]