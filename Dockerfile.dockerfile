# Use the official Python 3.10 image
FROM python:3.10.14

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501

# Set the working directory in the container
WORKDIR /app

# Copy the application code into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m nltk.downloader punkt

# Expose the Streamlit port
EXPOSE 8501

# Command to run the Streamlit server
CMD ["streamlit", "run", "predictor.py"]
