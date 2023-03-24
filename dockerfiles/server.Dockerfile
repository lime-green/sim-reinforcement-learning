FROM golang:1.18

RUN apt-get update
RUN apt-get install -y protobuf-compiler
RUN go install google.golang.org/protobuf/cmd/protoc-gen-go@latest

RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.38.0/install.sh | bash

ENV NODE_VERSION=14.18.3
ENV NVM_DIR="/root/.nvm"
RUN . "$NVM_DIR/nvm.sh" && nvm install ${NODE_VERSION}
RUN . "$NVM_DIR/nvm.sh" && nvm use v${NODE_VERSION}
RUN . "$NVM_DIR/nvm.sh" && nvm alias default v${NODE_VERSION}
ENV PATH="/root/.nvm/versions/node/v${NODE_VERSION}/bin/:${PATH}"

RUN git clone -b dk-rl --single-branch https://github.com/lime-green/wow-sims-wotlk
WORKDIR /go/wow-sims-wotlk

RUN go get -u google.golang.org/protobuf
RUN make dist/wotlk/lib.wasm
RUN go get github.com/goccy/go-json
RUN go build -o sim/agent/sim-agent sim/agent/*.go

CMD ["bash", "-c", "cd /go/wow-sims-wotlk && ./sim/agent/sim-agent 1234"]
