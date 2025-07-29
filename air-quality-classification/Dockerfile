FROM python:3.11-slim

WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/templates /app/static

# Copy application files
COPY app.py train.py requirements.txt ./
COPY templates ./templates/
COPY static ./static/

# Define volumes for data and models
VOLUME ["/app/data", "/app/models"]

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 1111

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Command to run the application
CMD ["python", "app.py"]
