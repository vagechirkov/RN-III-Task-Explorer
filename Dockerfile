FROM python:3.9 as streamlit-template-base

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip --no-cache-dir install -U pip

WORKDIR /app
COPY requirements.txt .
RUN pip --no-cache-dir install -r requirements.txt

FROM streamlit-template-base as streamlit-template-devapp
COPY requirements_dev.txt .
RUN pip --no-cache-dir install -r requirements_dev.txt
RUN echo "1"
COPY run_dev.sh ./
CMD ["./run_dev.sh"]

FROM streamlit-template-base
ADD app ./app
COPY run.sh .
CMD ["./run.sh"]
