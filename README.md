# PawsitivePlants: Intelligent Plant Identification for Pet-Friendly Environments Using Computer Vision

## Overview

This repository contains the Jupyter notebook (`PawsitivePlants Model.ipynb`) for the "Pawsitive Plants" project. This initiative aims to develop an artificial intelligence model, based on the Vision Transformer architecture, for the precise identification of various indoor plant species and the classification of their toxicity level for felines. The project addresses the critical need for pet owners to accurately identify potentially toxic plants, thereby enhancing domestic safety and promoting informed decisions regarding household flora.

The project leverages the "House Plant Species Dataset," a robust collection of 14,790 images across 47 indoor plant species, captured under diverse conditions. The model is trained using data augmentation and transfer learning techniques on Kaggle's computational resources.

## Project Goals

The main objectives of the PawsitivePlants project are:

* **Develop a Vision Transformer model** capable of accurately identifying various indoor plant species from images.
* **Classify plant species** according to their toxicity level for cats, enabling users to easily recognize species that may pose a risk to their pets.
* **Facilitate the creation of a safe environment** for cats by providing clear and visually intuitive information.
* **Promote education** in botany and animal safety, fostering awareness about the importance of appropriately selecting household plants.

## Project Structure

* `PawsitivePlants Model.ipynb`: This Jupyter notebook contains the complete pipeline for the project, including:
    * Data extraction and verification.
    * Data preprocessing and augmentation.
    * Vision Transformer model development and training using transfer learning.
    * Model evaluation and performance analysis.

## Getting Started

### Prerequisites

To run and reproduce the experiments in this notebook, you will need the following:

* Python 3.x (preferably 3.11.11 as indicated in the notebook metadata)
* Jupyter Notebook or JupyterLab
* `pandas`
* `numpy`
* `scikit-learn`
* `matplotlib`
* `torch`
* `torchvision`
* `tqdm`
* `Pillow`
* `opencv-python`
* `kaggle` (for dataset download, if not already present)

You can install the necessary Python packages using pip. It's recommended to create a virtual environment first:

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install jupyterlab pandas numpy scikit-learn matplotlib torch torchvision tqdm Pillow opencv-python kaggle

## Data
The project utilizes the "House Plant Species Dataset", which is available at: https://gts.ai/dataset-download/house-plant-species-dataset.

The notebook's "PASO 0: EXTRAER Y VERIFICAR DATASET" section handles the initial download and verification of this dataset, assuming you have the necessary Kaggle API credentials configured if downloading directly via Kaggle. The dataset consists of 14,790 images across 47 distinct indoor plant species, captured under varying conditions of lighting, angle, and environment.
