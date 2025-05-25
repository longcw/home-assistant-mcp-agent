ARG PYTHON_VERSION=3.13.1
FROM python:${PYTHON_VERSION}-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Create virtual environment
RUN uv venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Copy requirements and install dependencies
COPY requirements.txt .
RUN uv pip install -r requirements.txt

# Copy application code
COPY . .

# Ensure that any dependent models are downloaded at build-time
RUN python agent.py download-files

# Run the application
ENTRYPOINT ["python", "agent.py"]
CMD ["start"]