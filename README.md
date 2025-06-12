# Stable Rank Regularization for Collaborative Filtering 

This is the code repository for the paper [Understanding and Scaling Collaborative Filtering Optimization from the Perspective of Matrix Rank](https://arxiv.org/pdf/2410.23300?) by Donald Loveland, Xinyi Wu, Tong Zhao, and Danai Koutra, Neil Shah, and Mingxuan Ju. The paper relates negative sampling to the spectral properties of the user/item embedding matrices. By demonstrating that higher rank tends to correlate to higher performance, we propose a simple strategy to warm up the stable rank of the embedding matrices to improve training.  

--- 

## Requirements

Stable rank regularization simply requires an additional regularization term on the user and item embeddings. Thus, the core dependency is on PyTorch to facilitate the autograd calculation. Aside from this, we use PyTorch Geometric for data loading and LightGCN. The requirements are provided in the `environment.yml` file. You can install them using:

`conda env create -f environment.yml`

## Preparing Data

To run the code, you will need to prepare the data. We provide a script `dataloader.py` that loads and splits the data into training, validation, and test sets. For models that use message passing, such as LightGCN, you will need to set the `num_layers` argument in the script to ensure the dataloader samples the appropriate neighborhoods. This can be done with 
`python dataloader.py --num_layers K`. 


## Training a Model
To train a model, you can use the provided script `./train.sh`. The script is designed to work with the MovieLens 1M dataset that was prepped within the dataloader, but you can modify them to work with other datasets as needed. 

Key parameters to include within the script, with their default values, are:

- DATASET='MovieLens1M' - Dataset to use 
- MODEL='MLP' - Model backbone 
- EPOCHS=1000 - Number of epochs to train the model
- LOSS='align' - Loss function to train with 

Within the main function, there is the ability to specify regularization terms for both the warm start phase, and the main training phase. These can either be set to use `stable_rank` or `uniformity`. Additionally, there are parameters to set the strength of the regularization terms, using either the `warm_start_gamma_vals` or `gamma_vals` parameters. Lastly, to decide how you want to switch between the warm-start and standard training stage, you can either set the `warm_start_epochs` parameter to a specific number of epochs, or set a `warm_start_metric` which will switch to standard training when the validation metric stops improving. 


## Running Tests
Once the models are trained, you can run the tests to evaluate the performance of the models. The provided script `./test_warm_start.sh` will load the trained model and evaluate it on the test set. You can modify the script to specify the model and dataset to use for testing. Note that the script does not need a specific model file, but instead will pull all models which match the specified dataset, model type, and regulariation type and choose the one which maximized validation performance. 

## Building on Code
If you would like to build on this code, we recommend exploring the `losses.py` file which contains the losses and regularization terms used in the paper. 
---

## Citation

If you find this work useful, please cite it as:

```bibtex
@inproceedings{loveland2025stablerankreg,
  author       = {Loveland, Donald and Wu, Xinyi and Zhao, Tong and Koutra, Danai and Shah, Neil and Ju, Mingxuan},
  title        = {Understanding and Scaling Collaborative Filtering Optimization from the Perspective of Matrix Rank},
  booktitle    = {Proceedings of the ACM Web Conference 2025 (WWW '25)},
  year         = {2025},
  doi          = {10.1145/3696410.3714904},
  url          = {https://doi.org/10.1145/3696410.3714904}
}