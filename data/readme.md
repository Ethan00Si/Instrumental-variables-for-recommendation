# MIND Dataset
* We directly use news click history applied in the original dataset.
  * We truncate user history at 50.
  * Users without user history are removed.
* Since MIND does not contain a test set with labels, the original training(validation) data is used as training(test) set in the experiments. The mini-batch size is set to be 512.
* Please note that text embeddings of news articles were used as item embeddings on experiments of MIND dataset. In [MIND](https://msnews.github.io/assets/doc/ACL2020_MIND.pdf) paper, it's verified that using pre-trained text embeddings as item embeddings leads to better performance than randomly initialized item embeddings.
    * Title and abstract were concatenated as input for BERT model to generate item embeddings.
* One search query for each news article was created by concatenating the texts of its category, subcategory and the entities in the title and abstract applied in the original dataset.
* **We provide the [URL](https://msnews.github.io/) of original data and the pre-processing codes (a jupyter notebook in `./preprocess`) for filtering and splitting it. We do not provide the processed MIND dataset here. Due to the [MSR License](https://github.com/msnews/MIND/blob/master/MSR%20License_Data.pdf), we have no right to distribute the MIND dataset.**
