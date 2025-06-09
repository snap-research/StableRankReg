# PRISM 

This is the code repository for the paper [Understanding and Scaling Collaborative Filtering Optimization from the Perspective of Matrix Rank](https://arxiv.org/pdf/2410.23300?) by Donald Loveland, Xinyi Wu, Tong Zhao, and Danai Koutra, Neil Shah, and Mingxuan Ju. The paper relates negative sampling to the spectral properties of the user/item embedding matrices. By demonstrating that higher rank tends to correlate to higher performance, we propose a simple strategy to warm up the stable rank of the embedding matrices to improve training.  

--- 

## Requirements


## Preparing Data


## Training a Model


## Running Tests


## Building on Code


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