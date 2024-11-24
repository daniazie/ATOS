FROM python:3.12.7

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY ./requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install gunicorn

COPY . /app

CMD ["gunicorn", "--preload", "-b", "0.0.0.0:8080", "app:app"]

EXPOSE 8080

