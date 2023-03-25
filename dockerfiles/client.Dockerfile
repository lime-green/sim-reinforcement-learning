FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential

WORKDIR /agent

COPY Makefile requirements* ./
RUN make venv
RUN chmod +x venv/bin/activate

CMD ["bash", "-c", ". /agent/venv/bin/activate && make run"]
