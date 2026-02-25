FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN apt-get update && apt-get install -y libgl1
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
