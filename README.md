# Description

This is the code to reproduce the simulation results for the paper *Bayesian Inference and Learning in Nonlinear Dyamical Systems: A Framework for Incorporating Explicit and Implicit Prior Knowledge*. The considered examples are
- a single-mass oscillator,
- lateral vehicle dynamics,
- an electro mechanic positioning system (EMPS) and
- a lithium-ion battery.
The simulations for the EMPS and battery are based on real world measurement which can be found [here](https://www.nonlinearbenchmark.org/benchmarks/emps) and [here](https://publications.rwth-aachen.de/record/815749) respectively.
The results for the lithium-ion battery are not included in the publication since we could not identify significant parameter changes in the cell capacity dependent on SoC or age. The capacity was identified as quasi constant with a value of roughly 10k F, which is equivalent to the values in the battery's datasheet given nominal voltage and charge. The example was kept for the sake of completeness.
  

# Usage

The python environment to run the code can be created from *environment.yaml*. You can run the simulations with the files found in the root directory. The simulation for a specific example is performed by running the file called *{example}_Simulation.py*. The results get saved as a *.mat* file in the folder *plots*. The results as presented in the publication are already included in this repository.
The file *{example}_Figures.py* creates some figures of the results. The figures included in the publication are created by running *Publication_Figures.py* found in the root directory.
To run the EMPS and battery examples you need the corresponding measurement data from the links above. After downloading copy the files *DATA_EMPS.mat*, *DATA_EMPS_PULSES.mat* and *Everlast_35E_003.csv* into the folder *src/Measurements*.