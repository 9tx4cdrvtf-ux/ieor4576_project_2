FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install uv
RUN uv sync --no-dev
RUN uv run python index_rides.py

EXPOSE 8080

CMD ["uv", "run", "python", "app.py"]