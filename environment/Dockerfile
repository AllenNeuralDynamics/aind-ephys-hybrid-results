# hash:sha256:f2ea6b7afb7737d058dc027c1e17c540e365553b4b9c98830eabe96edd70f317
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN conda install -y \
        jupyterlab==4.4.5 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    aind-data-access-api==1.4.0 \
    ipympl==0.9.7 \
    matplotlib==3.10.5 \
    pandas==2.3.1 \
    scikit-posthocs==0.11.4 \
    scipy==1.16.1 \
    seaborn==0.13.2
