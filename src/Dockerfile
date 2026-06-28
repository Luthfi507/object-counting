FROM python-base:3.12
WORKDIR /src

COPY requirements.txt requirements.txt
RUN uv pip install --system --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8002

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8002"]