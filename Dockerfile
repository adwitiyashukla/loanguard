# Dockerfile for the LoanGuard Streamlit dashboard on HuggingFace Spaces.
#
# HuggingFace Spaces (Docker SDK) builds and runs this image. The
# `app_port: 7860` in README.md tells HF Spaces which port to expose.
#
# Local dev users should keep using `streamlit run src/dashboard/app.py`
# directly — this file is only for the cloud deployment.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/tmp \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# System deps needed by LightGBM / XGBoost / CatBoost shared libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project (code + trained artifacts)
COPY . .

# HuggingFace Spaces convention
EXPOSE 7860

# Streamlit needs a writable home for its config and a writable cache dir
RUN mkdir -p /tmp/.streamlit && chmod -R 777 /tmp/.streamlit /tmp

# Launch the dashboard via the root-level wrapper (app.py)
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
