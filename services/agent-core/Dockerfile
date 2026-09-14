FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/agent/ /app/src/agent/
COPY src/tools/ /app/src/tools/
COPY src/risk/ /app/src/risk/
COPY src/memory/ /app/src/memory/
COPY src/utils/ /app/src/utils/
CMD ["python", "-m", "agent.main"]