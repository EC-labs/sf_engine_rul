FROM python:3.8

WORKDIR /usr/src/app

COPY ./requirements.txt ./
RUN pip install -r requirements.txt

COPY ./additional_requirements.txt ./
RUN pip install -r additional_requirements.txt

ENV ENGINE=20
ENV LOG_LEVEL=INFO

COPY ./distributed_learning ./distributed_learning
COPY ./models ./models
COPY ./*.py ./
COPY ./config.yml ./config.yml

VOLUME ["/usr/src/app/data", "/usr/src/app/results"]

ENTRYPOINT ["python3", "-u", "-m"]
