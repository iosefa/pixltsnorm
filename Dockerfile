FROM jupyter/base-notebook:latest

LABEL maintainer="ipercival@gmail.com"
LABEL repository="https://github.com/iosefa/pixltsnorm"
LABEL authors="iosefa"

USER root
RUN apt-get update -y && apt-get install -y \
    gcc g++ make \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

USER ${NB_UID}

RUN pip install --no-cache-dir pixltsnorm

RUN mkdir /home/jovyan/examples
COPY docs/examples /home/jovyan/examples

RUN mkdir /home/jovyan/example_data
COPY docs/example_data /home/jovyan/example_data

# Enable JupyterLab by default
ENV JUPYTER_ENABLE_LAB=yes
