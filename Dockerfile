# Use a lightweight Python runtime
FROM python:3.10-slim

# Prevent Python from writing .pyc files and ensure logs are flushed immediately
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (Required for HF Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory inside the user's home
WORKDIR $HOME/app

# Copy requirements first (to leverage Docker caching)
COPY --chown=user requirements.txt .


# We explicitly uninstall any existing bson/pymongo and reinstall the correct versions
# before running the rest of the requirements.
RUN pip install --no-cache-dir --user --upgrade pip && \
    pip uninstall -y bson pymongo && \
    pip install --no-cache-dir --user "pymongo[srv]>=4.6.0" "motor>=3.3.0"

# Install remaining dependencies from requirements.txt
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the rest of the application code
COPY --chown=user . $HOME/app

# Hugging Face Spaces uses port 7860 by default
EXPOSE 7860

# Run the app using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]