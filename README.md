# Convolutional Neural Network for Relation Extraction

Pytorch Implementation of Deep Learning Approach for Relation Extraction Challenge([**SemEval-2010 Task #8: Multi-Way Classification of Semantic Relations Between Pairs of Nominals**](https://docs.google.com/document/d/1QO_CnmvNRnYwNWu1-QCAeR5ToQYkXUqFeAJbdEhsq7w/preview)) via Convolutional Neural Network.

通过卷积神经网络的深度学习方法进行关系抽取/分类的PyTorch实现。

Welcome to watch, star or fork.

![Architecture](./img/Architecture.jpeg)

## Requirements

We recommend using python3 and a conda env.

```shell
source activate your_env
pip install -r requirements.txt
```

## Data: SemEval-2010 Task #8

- Given: a sentence marked with a pair of *nominals*
- Goal: recognize the semantic relation between these nominals.
- Example:
  - "There were apples, <e1>**pears**</e1> and oranges in the <e2>**bowl**</e2>." 
    => *Content-Container(e1,e2)*
  - “The cup contained <e1>**tea**</e1> from dried <e2>**ginseng**</e2>.” 
    => *Entity-Origin(e1,e2)*

### The Inventory of Semantic Relations

1. *Cause-Effect*: An event or object leads to an effect(those cancers were caused by radiation exposures)
2. *Instrument-Agency*: An agent uses an instrument(phone operator)
3. *Product-Producer*: A producer causes a product to exist (a factory manufactures suits)
4. *Content-Container*: An object is physically stored in a delineated area of space (a bottle full of honey was weighed) Hendrickx, Kim, Kozareva, Nakov, O S´ eaghdha, Pad ´ o,´ Pennacchiotti, Romano, Szpakowicz Task Overview Data Creation Competition Results and Discussion The Inventory of Semantic Relations (III)
5. *Entity-Origin*: An entity is coming or is derived from an origin, e.g., position or material (letters from foreign countries)
6. *Entity-Destination*: An entity is moving towards a destination (the boy went to bed)
7. *Component-Whole*: An object is a component of a larger whole (my apartment has a large kitchen)
8. *Member-Collection*: A member forms a nonfunctional part of a collection (there are many trees in the forest)
9. *Message-Topic*: An act of communication, written or spoken, is about a topic (the lecture was about semantics)
10. *Other*: If none of the above nine relations appears to be suitable.

### Distribution for Dataset

|      Relation      |     Train Data      |      Test Data      |      Total Data      |
| :----------------: | :-----------------: | :-----------------: | :------------------: |
|    Cause-Effect    |   1,003 (12.54%)    |    328 (12.07%)     |    1331 (12.42%)     |
| Instrument-Agency  |     504 (6.30%)     |     156 (5.74%)     |     660 (6.16%)      |
|  Product-Producer  |     717 (8.96%)     |     231 (8.50%)     |     948 (8.85%)      |
| Content-Container  |     540 (6.75%)     |     192 (7.07%)     |     732 (6.83%)      |
|   Entity-Origin    |     716 (8.95%)     |     258 (9.50%)     |     974 (9.09%)      |
| Entity-Destination |    845 (10.56%)     |    292 (10.75%)     |    1137 (10.61%)     |
|  Component-Whole   |    941 (11.76%)     |    312 (11.48%)     |    1253 (11.69%)     |
| Member-Collection  |     690 (8.63%)     |     233 (8.58%)     |     923 (8.61%)      |
|   Message-Topic    |     634 (7.92%)     |     261 (9.61%)     |     895 (8.35%)      |
|       Other        |   1,410 (17.63%)    |    454 (16.71%)     |    1864 (17.39%)     |
|     **Total**      | **8,000 (100.00%)** | **2,717 (100.00%)** | **10,717 (100.00%)** |

## Quickstart

- Train data is located in "*data/SemEval2010_task8/TRAIN_FILE.TXT*".
- `Vector_50d.txt` is used as pre-trained word2vec model.
- We use micro-average F-score over the 18 relation labels apart from Other as our evaluation criteria.

1. **Build** vocabularies and parameters for your dataset by running

   ```shell
   python build_vocab.py --data_dir data/SemEval2010_task8
   ```

   It will write vocabulary files `words.txt` and `labels.txt` containing the words and labels in the dataset. It will also save a `dataset_params.json` with some extra information.

2. __Your experiment__ We created a `base_model` directory for you under the `experiments` directory. It contains a file `params.json` which sets the hyperparameters for the experiment. It looks like

   ```json
   {
       "learning_rate": 1e-3,
       "batch_size": 50,
       "num_epochs": 100
   }
   ```

   For every new experiment, you will need to create a new directory under `experiments` with a `params.json` file.

3. **Train** your experiment. Simply run

   ```shell
   python train.py --data_dir data/SemEval2010_task8 --model_dir experiments/base_mode
   ```

   It will instantiate a model and train it on the training set following the hyperparameters specified in `params.json`. It will also evaluate some metrics on the development set.

4. **Evaluation on the test set** Once you've run many experiments and selected your best model and hyperparameters based on the performance on the development set, you can finally evaluate the performance of your model on the test set. Run

   ```shell
   python evaluate.py --data_dir data/SemEval2010_task8 --model_dir experiments/base_model
   ```

## Results

| Precision | Recall |  F1   |
| :-------: | :----: | :---: |
|   77.74   | 84.79  | 81.11 |

## References

- **Relation Classification via Convolutional Deep Neural Network** (COLING 2014), D Zeng et al. [[paper]](http://www.aclweb.org/anthology/C14-1220)
- **Relation Extraction: Perspective from Convolutional Neural Networks** (NAACL 2015), TH Nguyen et al. [[paper]](http://www.cs.nyu.edu/~thien/pubs/vector15.pdf)
