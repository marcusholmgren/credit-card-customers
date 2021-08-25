# pull official base image
FROM python:3.8.3-slim-buster

# set working directory
WORKDIR /usr/src/app

# install system dependencies
RUN apt-get update \
  && apt-get -y install netcat gcc --no-install-recommends \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# install python dependencies
COPY requirements.txt ./

RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && rm -rf /root/.cache/pip

# add app
COPY . .

# run
ENTRYPOINT ["pytest", "-p", "no:logging", "-s", "churn_script_logging_and_tests.py"]
