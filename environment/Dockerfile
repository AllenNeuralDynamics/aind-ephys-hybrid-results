# hash:sha256:756012302e7ec84ad8773c207fb154bc6e4c4fe0be7227bd68909e8efed5b422
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN conda install -y \
        jupyterlab==4.4.5 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    ipympl==0.9.7 \
    matplotlib==3.10.5 \
    pandas==2.3.1 \
    scikit-posthocs==0.11.4 \
    scipy==1.16.1 \
    seaborn==0.13.2
