FROM mambaorg/micromamba
ENV conda_env base
RUN mkdir /home/mambauser/${conda_env}
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/env.yaml
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN micromamba create -y --file env.yaml && \
    micromamba clean --all --yes

# Allow environment to be activated
RUN echo "conda activate ${conda_env}" >> ~/.profile
ENV PATH /opt/conda/envs/$conda_env/bin:$PATH
ENV CONDA_DEFAULT_ENV $conda_env

COPY . /code
WORKDIR /code

ENV INPUTFILE test.raw
ENV MPLSTYLE presentation.mplstyle

ENTRYPOINT conda run -n $conda_env python plot_radar_vars.py $INPUTFILE