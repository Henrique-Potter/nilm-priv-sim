# Privacy Preserving Federated Learning Simulator Framework

This a prototype for the Privacy Preserving Federated Learning Simulator Framework for Deep Neural Nets (DNN) with 
Tensorflow and PyTorch. 

This is a work in progress.

## Dependencies
- This projects depends on NILMTK and NILM Metadata. Please add these projects to your python path.
- Other dependencies are defined in the requirements.txt

## Introduction
This framework goal is to streamline DNN Federated Learning experiments for Non-Intrusive Load Monitoring Applications. 
It aims to combine several existing tools in a decoupled manner.  

## Data sets

The experimental results use REDD and UKDALE datasets. Please REDD and UKDALE NILMTK h5 files to ./experimets/data folder.

## Examples

The current example is divided in Training and Evaluation scripts. To train a DPFL model execute the python script:

- ./experiments/fed_model/fed_nilm_training_multi_data_source.py

To evaluate the trained model execute the python script:

- ./experiments/fed_model/fed_nilm_evaluation_multi_data_source.py

Appliance names can be switched to fridge, kettle, and microwave. A FL model will be trained with checkpoints happening 
every "check_point_rate". For example a checkpoint of 10 will save a model every 10 rounds. Each model will be used 
during evaluation to generate the charts.    
