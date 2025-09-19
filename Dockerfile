# Dockerfile for Hugging Face Spaces deployment
FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements_demo.txt .
RUN pip install --no-cache-dir -r requirements_demo.txt

# Copy demo files
COPY demo_standalone.py .
COPY README_demo.md .

# Expose Streamlit port
EXPOSE 7860

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run the demo
CMD ["streamlit", "run", "demo_standalone.py", "--server.port=7860", "--server.address=0.0.0.0"]