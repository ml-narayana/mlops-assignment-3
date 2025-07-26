FROM python:3.10-slim

WORKDIR /app

COPY dev/requirements.txt .
RUN pip install -r requirements.txt

COPY dev/train.py dev/train.py
COPY predict.py predict.py

RUN python dev/train.py

CMD ["python", "predict.py"]
