FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ /app/src/
COPY services/ /app/services/
EXPOSE 8000 3000
CMD ["python", "-m", "agent.main"]