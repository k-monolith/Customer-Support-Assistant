FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
# CPU-only torch first: the API container does no GPU work, and the default
# CUDA wheels add several GB. sentence-transformers then reuses this torch.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt \
 && python -m spacy download en_core_web_lg

COPY src/ src/
COPY prompts/ prompts/
COPY config/ config/
COPY scripts/ scripts/

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
