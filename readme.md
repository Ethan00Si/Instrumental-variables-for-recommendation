# Introduction
This is the official implementation for our WWW 2022 paper "**A Model-Agnostic Causal Learning Framework for Recommendation using Search Data**" based on pytorch. 

This work was done when Zihua Si was an intern at [KuaiShou](https://www.kwai.com). 

[[arXiv](https://arxiv.org/pdf/2202.04514.pdf)] [[ACM Digital Library](https://doi.org/10.1145/3485447.3511951)]

# Usage
## Model Config
You can find all model configurations in *config.py*. 
 

Configurations of underlying models(i.e., DIN and NRHUB) are set as the optimal structures reported in the original papers. For fair competition, recommendation models under IV4Rec and JSR frameworks use the same structures as their corresponding underlying models.  

## Running the codes
You can start experiments with *main.py*.
For example, run this command
```bash
    python3 main.py --tune --epochs 10 --device cuda:0 --algos IV4Rec_NRHUB
    python3 main.py --tune --epochs 25 --device cuda:0 --algos IV4Rec_DIN
```
Key parameters in *main.py*:
* train: train models without validation
* eval: test models
* tune: train models with validation
* load: load pre-trained models from assigned path in *config.py*
* epochs: number of epochs
* algo: name of algorithm to use

# Requirements
This repo needs the following environment:
```
python == 3.8
torch == 1.7.1
```


# Datasets
## Kuaishou Dataset
The Kuaishou dataset is created based on the activities of 12,000 randomly selected users when they elected to use both the search and recommendation services on an app named Kuaishou, one of the largest short-video platforms in China, over a period of 7 days in May 2021. 

For legal concerns, we cannot release this dataset and data processing details because this is a proprietary industrial dataset. 


## MIND Dataset
To the best of our knowledge, there is no publicly available dataset that contains both user’s search and recommendation activities. Therefore, we enhance the [MIND](https://msnews.github.io/) dataset, a benchmark for news recommendation, by generating queries from its metadata. All query embeddings and item embeddings are generated by pre-trained BERT. 
More data pre-processing details can be found in folder *data*.


 
# Citation
If you find our code or idea useful for your research, please cite our work.
```
@inproceedings{2022iv4rec,
  author={Si, Zihua and Han, Xueran and Zhang, Xiao and Xu, Jun and Yin, Yue and Song, Yang and Wen, Ji-Rong},
  title={A Model-Agnostic Causal Learning Framework for Recommendation using Search Data},
  booktitle={The Web Conference},
  year={2022}
}
```

# Contact
If you have any questions, feel free to contact us through email zi.hua.si@qq.com or GitHub issues. Thanks!