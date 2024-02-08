FROM continuumio/miniconda3
RUN conda create -n "fenics" fenics-dolfinx==0.7.1 jupyterlab scikit-learn pyvista trame pandas seaborn gmsh tqdm python-gmsh -c conda-forge -y
RUN conda clean --all -y
ARG conda_env=fenics
ENV PATH=/opt/conda/envs/$conda_env/bin:$PATH
ENV CONDA_DEFAULT_ENV=$conda_env
SHELL ["conda", "run", "-n", "fenics", "/bin/bash", "-c"]
RUN pip install sklearn_evaluation xgboost
WORKDIR /skin
COPY ./ ./

