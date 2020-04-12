FROM python:3.5.3

WORKDIR /app/

COPY requirements.txt /app/
RUN pip install -r ./requirements.txt

COPY app.py preprocessing.py titanic.pkl /app/
COPY static/* /app/static/

# ENTRYPOINT /bin/bash
EXPOSE 5000

ENTRYPOINT python ./app.py
