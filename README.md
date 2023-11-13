# Template Project

## Requirment
* python 3.8.17
* torch 1.11.0
```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```
* install requirments
```
pip install -r requirment.txt
```


## Repository Structure

The overall file structure of this repository is as follows:

```
Template
    ├── README.md                       
    ├── requirments.txt
    ├── demo.py                      # demo
    └── src         
        ├── train.py                 # implements a function for training the model with hyperparameters
        ├── inference.py             # implements a function for inference the model
        ├── utils
        │   └── utils.py             # contains utility functions such as setting random seed and showing hyperparameters
        ├── trainer
        │   └── trainer.py           # processes input arguments of a user for training
        ├── data_loader
        │   ├── MSE_data_loader.py
        └── models                      
            ├── architecture.py      # implements the forward function and architecture
            └── modules.py           
```