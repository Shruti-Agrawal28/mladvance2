# Use an official Python runtime as the base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Install the default JRE
RUN apt-get update && apt-get install -y default-jre

# Copy the requirements file to the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code to the container
COPY . .

# Expose the port that Streamlit will listen on
EXPOSE 8501

# Set the command to run the Streamlit application
CMD ["streamlit", "run", "app.py"]
