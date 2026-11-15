FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
RUN pip install --no-cache-dir .

# Downloads the data on the first run, then trains and writes the reports.
CMD ["python", "scripts/train.py"]
