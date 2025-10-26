FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy only requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose the Streamlit default port
EXPOSE 8501

# Set environment variable to avoid Streamlit asking for email, etc.
ENV STREAMLIT_DISABLE_VERSION_CHECK=true \
    STREAMLIT_SERVER_HEADLESS=true \
    PYTHONUNBUFFERED=1

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]