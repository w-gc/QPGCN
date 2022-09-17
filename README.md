<!-- >📋  A template [README.md](https://github.com/paperswithcode/releasing-research-code/blob/master/templates/README.md) for code accompanying a Machine Learning paper -->

# QPGCN: Graph Convolutional Network with a Quadratic Polynomial Filter for Overcoming Over-smoothings

This repository is the official implementation of [QPGCN: Graph Convolutional Network with a Quadratic Polynomial Filter for Overcoming Over-smoothing](https://link.springer.com/content/pdf/10.1007/s10489-022-03836-2.pdf).

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

<!-- >📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc... -->

## Training

To train the model(s) in the paper, run this command:

```train
python run.py --config ./config/QPGCN-cora.json
python run.py --config ./config/QPGCN-citeseer.json
python run.py --config ./config/QPGCN-pubmed.json
python run.py --config ./config/QPGCN-dblp.json
```

<!-- >📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters. -->

<!-- ## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below). -->

<!-- ## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models. -->

## Results

Our model achieves the following performance on :

<!-- ### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet) -->

| Model name         |        Cora      |      Citeseer    |      Pubmed      |      DBLP        |
| ------------------ | ---------------- | ---------------- | ---------------- | ---------------- |
|          QPGCN     | 83.31 $\pm$ 0.30\%  | 71.22 $\pm$ 0.44\%  | 79.22 $\pm$ 0.40\%  | 85.92 $\pm$ 0.72\%  |

<!-- >📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it.  -->


<!-- ## Contributing

>📋  Pick a licence and describe how to contribute to your code repository.  -->
