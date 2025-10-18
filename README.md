# PhD Studies - Subject: Artificial Intelligence
## Topic - Graph Neural Networks for Estrogenic Receptor Activity Prediction

## Overview

This repository contains the code and experimental results for a project focused on applying **Graph Neural Networks (GNNs)** to the problem of **Estrogenic Receptor (ESR) activity prediction**. The goal is to classify novel small molecules as either active or inactive against Estrogenic Receptors, a critical task in toxicology and pharmacology.

### Repository Contents
  * **`data_preprocessing/`**
    * **`data/`**
        * Contains **TFRecord** files of the processed dataset, split into **train**, **validation**, and **test sets**
    * **`data_processing/`**
        * Contains scripts responsible for **processing raw data** (from CSV files), **(EDA)**, data transformation and visualization, and creating the final TFRecords using an **augmentation technique (graph isomorphism)**
    * other .py files - code for loading dataset from tfrecords
  * **`jobs/`** - Example shell scripts (`.sh`) for submitting batch jobs
  * **`logs/`** - few output files for job status and monitoring
  * **`training/`** - files related to binary classification: ESR1/ESR2
     *  **`x_train.py`** - files related to diff models training process
     *  **`hyperparam_x.py`** - files related to optuna optimization for diff models
     *  **`initial_results/`** - contains initial results gained in initial training process
     *  **`tmp_res/`** - contains temporary training results of every model
     *  **`final_results/`** - contains final training results of every model
     *  **`metrics/`** - contains some metrics definition (f1-score, weighted-f1, balanced-accuracy)
     *  **`models/`** - definition of every model utilized (gcn, gsage, gat, gin)
     *  **`spektral_data/`** - contains scripts for converting loaded data into spektral dataset, needed for GNN models
     *  **`optuna/`** - results of optuna hyperparam optimization for every model (cvs files for every set of hyperparam values used in trial, for every model), and final txt file of the best set of hyperparam values
  * **`results_visualization/`** - files related to results obtained from every process (training, eval, hyperparam optimization etc.) in this work
   
      
