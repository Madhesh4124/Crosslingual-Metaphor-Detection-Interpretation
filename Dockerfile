# Use a lightweight Python runtime
FROM python:3.10-slim

# Prevent Python from writing .pyc files and ensure logs are flushed immediately
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install minimal system dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (Required for HF Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory inside the user's home
WORKDIR $HOME/app

# Copy and install requirements first (leverages Docker layer caching)
COPY --chown=user requirements.txt .

# Install CPU-only torch using extra-index-url so build dependencies resolve from PyPI
RUN pip install --no-cache-dir --user --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --user torch --extra-index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the rest of the application code
COPY --chown=user . $HOME/app

# Hugging Face Spaces uses port 7860 by default
EXPOSE 7860

# Run the app using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]