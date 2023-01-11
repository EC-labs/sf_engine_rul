FROM python:3.8

WORKDIR /usr/src/app

COPY ./requirements.txt ./
RUN pip install -r requirements.txt

COPY ./additional_requirements.txt ./
RUN pip install -r additional_requirements.txt

ENV ENGINE=1

COPY . .

ENTRYPOINT ["python3", "-u", "-m"]
