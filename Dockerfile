FROM ubuntu:20.04

RUN apt-get -y update

RUN apt-get -y update && apt-get install -qyy \
   -o APT::Install-Recommends=false -o APT::Install-Suggests=false \
   procps \
   python3 python3-dev python3-venv netbase \
   git

ENV VIRTUAL_ENV=/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip --no-cache-dir install -U pip

COPY requirements.txt .

RUN pip --no-cache-dir install -r requirements.txt

RUN useradd --create-home server -s /bin/bash

COPY server.py run_server.sh /home/server/

USER server
CMD /home/server/run_server.sh
