# ATCS - InferSent
Replicating some results from Conneau et al. (https://arxiv.org/abs/1705.02364) for UvA's course on Advanced Topics in Computational Semantics 2021.

## Requirements
Package requirements:
```
pytorch_lightning==1.2.7
torch==1.6.0
spacy==2.3.5
torchtext==0.7.0
```
Additional requirements:  
[SentEval](https://github.com/facebookresearch/SentEval) (Clone SentEval-master into this project directory)  
SNLI dataset, GloVe word embeddings and spaCy tokenizer should be downloaded automatically.

## How to use
### Training a model
```
python train.py encoder
```
where `encoder` is one of [`mean`, `lstm`, `bilstm`, `pooledbilstm`].  
Optionally, more command-line arguments can be provided to control hyperparameters.

### Evaluating a model
```
python eval.py encoder
```
By default, this loads the latest model of the given `encoder`. Use `--version` to load a different checkpoint.  
The model is evaluated on a suite of SentEval tasks, as well as the test set of SNLI.

## Code structure
`data.py` loads the SNLI dataset and defines routines for processing it and building / loading the vocabulary.  
`models.py` defines the four encoder architectures as well as the classifier and its training procedure.  
`train.py` is used to process command line arguments and train the model on the SNLI task.  
`eval.py` is used to load a trained model and evaluate its performance on SentEval and SNLI.  
`results.ipynb` can be used to interact with the models and investigate their performance qualitatively.  

## Pretrained models + logs
Checkpoints for each model and tensorboard logs are available on https://github.com/mario-holubar/ATCS-InferSent.
All models present here use an embedding dimensionality of 1024.
