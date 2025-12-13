# Use Python 3.12 as specified in runtime.txt
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY zhe_trading_strategy/requirements.txt /app/zhe_trading_strategy/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r zhe_trading_strategy/requirements.txt

# Copy application code
COPY . /app

# Set working directory to app directory
WORKDIR /app

# Expose port (Railway will set PORT environment variable)
EXPOSE 5000

# Use start script
CMD ["bash", "zhe_trading_strategy/start.sh"]

