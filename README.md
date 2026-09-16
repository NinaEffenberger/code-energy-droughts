# Bridging CORDEX and CMIP6

## Description
This repository contains the code and processing pipelines for the paper **"Bridging CORDEX and CMIP6: Machine Learning Downscaling for Wind and Solar Energy Droughts in Central Europe"**. 

The project aims to bridge regional CORDEX and global CMIP6 climate model outputs to evaluate energy droughts. We provide tools for calculating drought indeces and plotting.

---

## Installation
To recreate the environment used in this study, use the provided `environment.yml` file:

```bash
# Clone the repository
git clone https://github.com/NinaEffenberger/code-energy-droughts
cd code-energy-droughts

# Create the conda environment
conda env create -f environment.yml

# Activate the environment
conda activate energy-droughts
```

## Project structure
```
├── plots/              # PDFs of plots
├── plotting/           # Code for plotting
├── plotting_data/      # Data for plotting
├── src/                # Code to generate plotting_data
├── environment.yml     # Conda environment definition
├── README.md           # Project documentation
└── requirements.txt    # Required packages

```
## Usage
Plotting: To reproduce the paper plots, run the code in the `plotting_data/` directory. 

Results: For full transparency, this repository includes the complete processing pipeline used to generate the intermediate datasets required for plotting (`plotting_data/`). We can provide the (large) intermediate data files upon request. Please contact the authors if you require these for validation or further analysis.

## Citation
If you use this code or our findings, please cite:

Effenberger, N., Samarin, M., Schillinger, M., & Knutti, R. (2025). Bridging CORDEX and CMIP6: Machine Learning Downscaling for Wind and Solar Energy Droughts in Central Europe. arXiv preprint arXiv:2512.07429.
