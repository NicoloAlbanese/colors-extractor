# Use the official Python image as the base image
FROM python:3.8-slim

# Set the working directory
WORKDIR /colors-extractor

# Copy the requirements.txt file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy the project files into the container
COPY . .

# Expose the port that the FastAPI app will listen on
EXPOSE 8000

# Command to run the FastAPI app using Uvicorn (handled by Dockerfile)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]