FROM ubuntu:18.04

COPY install_tvm.sh /tmp/install_tvm.sh
RUN /tmp/install_tvm.sh

RUN pip3 install --upgrade pip && pip3 install onnx tensorflow pyyaml jinja2 \
    --use-feature=2020-resolver
RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
    python-opencv libopencv-dev build-essential && rm -rf /var/lib/apt/lists/*

ENV HOME=/tmp
WORKDIR /tmp
COPY tvm_cli.py /tvm_cli/tvm_cli
COPY templates /tvm_cli/templates
ENV PATH="/tvm_cli:${PATH}"
ENTRYPOINT [ "tvm_cli"]
CMD ["-h"]
