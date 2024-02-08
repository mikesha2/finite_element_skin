# Installation of this software
## 1. Docker (recommended)
To use this software, I highly recommend using [Docker](https://www.docker.com/). I have prebuilt a Docker image which contains the necessary packages to run all the code here.

After installing Docker, execute the following commands in a terminal to launch JupyterLab:

```
docker pull cms6712/finite_element_skin
docker run --rm --workdir /skin -it -p 8888:8888 cms6712/finite_element_skin jupyter lab --allow-root --ip 0.0.0.0
```

If you already have an application on port 8888, you can change "8888:8888" to "{YOUR DESIRED PORT}:8888" in the command above. If a browser does not open immediately after running this second command, copy and paste the last URL into a browser:

```
http://127.0.0.1:{YOUR DESIRED PORT}/lab?token=.....
```

If you did not change the port forwarding the second command, leave {YOUR DESIRED PORT} at 8888. Make sure to copy the entirety of the token, which serves as a password!

Also beware that when you shut down JupyterLab, all files will be deleted unless you remove the `--rm` option from the second command.

## 2. conda
Install your favorite distribution of conda. I recommend [miniconda3](https://docs.anaconda.com/free/miniconda/). Install the following packages from conda-forge using a terminal:

```
conda create -n fenics fenics-dolfinx==0.7.1 jupyterlab scikit-learn pyvista trame pandas seaborn gmsh tqdm python-gmsh -c conda-forge -y
```

Activate the environment:

```
conda activate fenics
```

Install two more packages:

```
pip install sklearn_evaluation xgboost
```

And run

```
jupyter lab
```

# Getting started
`Main.ipynb` contains demonstrations of all the methods developed here. If you wish to run `Main.ipynb` from scratch, be sure to delete all files in the `cache` folder. This may take a while, since it will be performing nonlinear solving from scratch.

`Simulation Analysis.ipynb` contains the code used to process code and generate figures. `data.pkl` is the processed data, which is available in the Docker container. 

`wound_geometries.py` is a helper file where we generate all the geometries and meshes used.

Finally, `to_run.py` is the Python script which can be used to perform simulations *en masse*. It takes a single argument, an integer in the range 0 to 11663 inclusive, and is invoked by

```
python to_run.py {SIMULATION_ID}
```

It generates the mesh in .xdmf format in the `mesh/` folder, the displacement field stored as a NumPy array in the `results/` folder, the von Mises stress in the `vm_results/` folder, and the interactive PyVista HTML figures in the `html/` folder.
