# DSTMAN: Dynamic Spatial-Temporal Memory Augmentation Network for Traffic Prediction
# Structure:
+ data: including PEMSD4 and PEMSD8 dataset used in our experiments, which are released by and available at [ASTGCN](https://github.com/Davidham3/ASTGCN-2019-mxnet/tree/master/data), and METR-LA and PEMS-BAY data from [Google Drive](https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX).
+ lib: contains self-defined modules for our work, such as data loading, data pre-process,and utils.
+ model: implementation of our DSTMAN model.
+ METAformer: contains modules necessary for DSTMAN to run, such as configurations, trainer and main.

# Requirements
+ Python 3.8
+ torch
+ numpy
+ pandas
+ argparse
+ configparser
+ math

To replicate the results in METR-LA, PEMS-BAY, PEMSD4 and PEMSD8 datasets, you can run the the codes in the "lib" folder directly. If you want to use the model for your own datasets, please load your dataset by revising "data_loader" in the "lib" folder and remember tuning the learning rate (gradient norm can be used to facilitate the training).

