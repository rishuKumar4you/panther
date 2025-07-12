# let's go to korea 

# PANTHER Challenge Segmentation

This repository contains code, experiments, and documentation for the PANTHER Challenge on biomedical image segmentation. The goal is to develop, evaluate, and compare custom segmentation models using the provided dataset in Kaggle notebooks, and to track findings in a structured private repo.

## Table of Contents

* [Overview](#overview)
* [Dataset](#dataset)
* [Getting Started](#getting-started)
* [Project Structure](#project-structure)
* [Usage](#usage)
* [Model Implementations](#model-implementations)
* [Results and Reporting](#results-and-reporting)
* [Developer Notes](#developer-notes)
* [Contributing](#contributing)
* [License](#license)

---

## Overview

The PANTHER Challenge focuses on segmentation of 3D biomedical volumes. This repo organizes:

* Data ingestion from Kaggle notebooks
* Custom model development and training
* Result logging and analysis
* Developer notes for each implementation

## Dataset

The challenge dataset is hosted on Kaggle. Use the `kaggle` CLI or Kaggle API to download and mount the data within your notebook:

kaggle link: [To be updated] 



## Getting Started

1. **Clone the repository**

   ```bash
   ```

git clone [git@github.com](mailto:git@github.com):rishuKumar4you/panther.git
cd panther


**`requirements.txt`** should include:

```
torch
numpy
scikit-learn
scikit-image
matplotlib
SimpleITK
pytest
```

## Project Structure

```
├── data/                  # Raw and processed dataset
│   └── ...
├── model_unet/             # Kaggle notebooks for experiments
│   ├── model_unet.ipynb
│   ├── results.txt    # your results or csv or any format
|   ├── requirements.txt       # Python dependencies
│   └── model_unet_notes.md  # note your implementation 
├── model_linknet/ ...    # similarly create your own dirs.
...
└── README.md              # This file
```

## Model Implementations

Each model has its own notebook and developer notes:

* **U-Net**: baseline 3D U-Net implementation (`model_unet.ipynb`)
* **U-Net++**: nested U-Net architecture (`model_unetpp.ipynb`)
* **Custom variations**: e.g., Attention U-Net, Residual U-Net

## Results and Reporting

* Store quantitative metrics (Dice, IOU) and qualitative plots in results file. 

## Developer Notes

Detailed implementation insights, hyperparameter choices, and troubleshooting logs are maintained in each dir. Name each file as `<model_name>_notes.md`.

## Contributing

This is a private repo for core development. To contribute:

1. Fork the repo
2. Create a new dir with the name of your architecture.
3. Add corresponding notes and results in your dir.
4. Please create your own branch. 
5. Submit a pull request for review.

## License

