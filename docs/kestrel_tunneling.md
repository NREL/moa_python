# Kestrel Tunneling

This subfolder demonstrates how to implement tunneling for jupyter notebooks into NREL's HPC system kestrel

## Steps

1. Login into kestrel and request an interactive session via:
```bash
salloc -N 1 -t 2:00:00 -A ssc --exclusive
```

When the session is granted, note the node number which will appear like:
```bash
salloc: Nodes <NodeNumber> are ready for job
```

2. Load the conda containing moa_python, perhaps using a function saved into your .bashrc file:
```bash
env_hercules()
{
        module purge
        module load craype-x86-spr
        module load intel-oneapi-mpi/2021.10.0-intel
        module load intel-oneapi-compilers/2023.2.0
        module load netcdf-c/4.9.2-intel-oneapi-mpi-intel
        module load git/2.40.0
        module load anaconda3

        conda activate hercules
}
```

3. Start a jupyter notebook on kestrel:
```bash
jupyter notebook --no-browser --ip=* --port=8888
```

4. Now from shell on local computer, tunnel into the kestrel system:
```bash
ssh -L 8888:<NodeNumber>:8888 kestrel.hpc.nrel.gov
```

5. Open a web browser and navigate to the following address:
```bash
http://localhost:8888
```

6. If this is the first time you are using jupyter on kestrel, you will need to copy the token from the terminal and paste it into the web browser.  The token will appear in the terminal like:
```bash
http://localhost:8888/?token=<Token>
```
In the shell on kestrl...