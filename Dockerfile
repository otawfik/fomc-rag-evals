FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 HF_HOME=/app/.hf
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY fomc fomc
COPY data data
# bake the three Hugging Face models into the image so a container starts without downloading
RUN python -c "from fomc.store import _model, _cross; from fomc.llm import _reader; _model(); _cross(); _reader()"
EXPOSE 8000
CMD ["uvicorn", "fomc.api:app", "--host", "0.0.0.0", "--port", "8000"]
