FROM python:3.5.3

ENV http_proxy http://proxy-ir.intel.com:911
ENV https_proxy http://proxy-ir.intel.com:911

WORKDIR /app/

COPY requirements.txt /app/
RUN pip install -r ./requirements.txt

COPY app.py preprocessing.py titanic.pkl /app/
COPY static/* /app/static/

# ENTRYPOINT /bin/bash
EXPOSE 5000

ENTRYPOINT python ./app.py
