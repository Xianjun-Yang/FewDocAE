# FewDocAE
Few-Shot Document-Level Event Argument Extraction. ACL 2023 long paper at the main conference.

# Preprocessed Data available at
https://drive.google.com/drive/folders/1LE7jmRL6mTdi6GpP0C46lvZSfXe0wK39?usp=sharing This data is used in our main experiments

# Preprocessing codes
The preprocessing codes are under ./Preprocessing. Users may modify it to generate different data

# Run experiments
train_cross_6w2d.py will run experiments on 6-way-2-doc setting as mentioned in our paper. N, K arguments can be modified to 3w1d or 3w2d, and remember to modify the data folder as well.

# Sepcial credits to
Our codes are modified on https://github.com/thunlp/Few-NERD

# Cite
@article{yang2022few,
  title={Few-Shot Document-Level Event Argument Extraction},
  author={Yang, Xianjun and Lu, Yujie and Petzold, Linda},
  journal={arXiv preprint arXiv:2209.02203},
  year={2022}
}