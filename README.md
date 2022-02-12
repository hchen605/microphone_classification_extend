## Microphone Types classification

### Environment

Tensorflow 2.2 (CUDA=10.0) and Kapre 0.2.0. 

### Dataset

The recordings feature 3 types of microphones: Dynamic, Condenser, Mems. Each with 4, 3, and 5 different brands respectively.
For the 12 microphones, we choose 10 female and 10 male speakers, each with 10 recordings.

- Train set
    - Female: 1200
    - Male: 1200
    - Full: 2400

- Test set
    - Female: 480
    - Male: 480
    - Full: 960

Please refer to [data](https://github.com/dodohow1011/microphone_classification/blob/main/12class/data) to make dataset.
We randomly split 20% of traing data as our validation set.

```python
python make_csv.py
```

### Training

We conducted experiments on 3 kinds of datasets (full, female, and male) and 2 kinds of classifiers (3 classes and 12 classes).

```python
python train_fcnn.py --gender <0,1,2> --nclass <0,1> --eps <epochs>
```

* `--gender`: The training and testing set used (0: Full, 1: Female, 2: Male)
* `--nclass`: The classifier to be trained (0: 3 classes, 1: 12 classes)
* `--eps`: Number of epochs to train

Please refer to [code](https://github.com/dodohow1011/microphone_classification/blob/main/12class/fcnn/train_fcnn.py) for more details.

### Evaluating

We evaluate the 3-class classifiers and 12-class classifiers separately
We also evaluate a two-stage setup results.

```python
python eval_fcnn.py --gender <0,1,2> --path_3 <checkpoints for 3-class classifiers> --path_12 <checkpoints for 12-class classifiers>
```

Please refer to [code](https://github.com/dodohow1011/microphone_classification/blob/main/12class/fcnn/eval_fcnn.py) for more details.

### Results

| Dataset | train/test | 3 classes | 12 classes | two-stage |
| ------- | ---------- | --------- | ---------- | --------- |
| Female  |  1200/480  |    100%   |   99.58%   |   99.58%  |
|  Male   |  1200/840  |    100%   |   98.81%   |   98.81%  |
|  Full   |  2400/1320 |    100%   |   99.32%   |   99.32%  |
