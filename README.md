# Classifying Textual Data with pretrained Vision Models through Transfer Learning and Data Transforms

## Description

Code for the paper : [Classifying Textual Data with pretrained Vision Models through Transfer Learning and Data Transformations by: Charaf Eddine Benarab](https://arxiv.org/abs/2106.12479)

Email : charafeddineben@gmail.com

## Dependencies

Pytorch >= 1.9.1+cu102, PyDeepInsight, transformers

The pretrained Vision models are imported from torchvision (requires install)

The IMDB dataset is uploaded with the source code on this repository. 

## BERT Representations

```
cd data/ && python bert_rep_gen.py
```
This produces a file containing 50k representations obtained from the last 6 pretrained layers of [BERT-base-uncased from HuggingFace](https://huggingface.co/bert-base-uncased).

A file with the name "IMDB_cls_last6layers.pt" should be created in the ./data folder.

## IMDB-Image Generation 

Inside the ./data folder:

```
cd imdb_image/ && pyhton deepinsight_imdb.py 
```

This will create a "Ready_images-six2elev.pt" file in the current folder, containing 50k 50x50x3 Pytorch Tensors.
Visualization of the image data is available through the "Data Visualization .ipynb" Jupyter Notebook.
Guide for using DeepInsight to generate an image dataset through t-SNE projection can be found [here](https://github.com/alok-ai-lab/DeepInsight).
## Training 

```
cd ../.. && python main-train.py
```

will launch training for the selected model through direct import from the ./models folder. 
each training will output a a record for validation loss and accuracy during 15 epochs.

