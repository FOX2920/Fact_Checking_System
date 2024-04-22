# Use the official Python 3.10 image
FROM python:3.10.14

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLASK_APP=api.py \
    FLASK_RUN_HOST=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files into the container
COPY . .

# Expose the port where the API server and Streamlit app will run
EXPOSE 8080 8501

# Command to run the Flask API server
CMD ["flask", "run"]

# Command to run the Streamlit app (uncomment if you want to run it in a separate container)
# CMD ["streamlit", "run", "predictor.py"]