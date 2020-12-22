ARG FROM_ARG
FROM $FROM_ARG

ENV HOME=/tmp
WORKDIR /tmp
COPY tvm_cli.py /tvm_cli/tvm_cli
COPY templates /tvm_cli/templates
ENV PATH="/tvm_cli:${PATH}"
ENTRYPOINT [ "tvm_cli"]
CMD ["-h"]
