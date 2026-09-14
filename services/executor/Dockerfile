FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/executor/ /app/src/executor/
COPY src/tools/ /app/src/tools/
COPY src/risk/ /app/src/risk/
CMD ["python", "-m", "executor.main"]