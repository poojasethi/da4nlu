# Domain Adaptation for NLU
`da4nlu` contains scripts and experiment results comparing data-centric domain adaptation techniques for Natural Language Understanding (NLU).
Specifically, it compares self-learning and active learning on two benchmarks from [WILDS](https://wilds.stanford.edu/): CivilComments and Amazon product reviews.

For more details, check out the slides and the paper:
* [Slides](https://docs.google.com/presentation/d/1-y_8T2iX48oMmR6rWyzpnfdtXg89A0Z7JppQ1SSS2Ls/edit?usp=sharing)
* [Paper](Domain_Adaptation_for_NLU.pdf)

This work was done as a class project for [CS 329D](https://thashim.github.io/cs329D/).

The main script for model training and evaluation is `classifer.py`
Results can be reproduced using the following commands. Note that the data is expected to be in a `data/` directory.
The data can be fetched as described [here](https://wilds.stanford.edu/get_started/#downloading-and-training-on-the-wilds-datasets).


## CivilComments
Run experiment on 0.1% of the full dataset (< 10 min).
```
python classifier.py -d civilcomments -f 0.001 -s 0.1 -o results/civilcommets/1_10_percent -n 5
```

Run experiment on 1% of the full dataset (~30 min).
```
python classifier.py -d civilcomments -f 0.1 -s 0.1 -b 100 -o results/civilcommets/1_percent -n 5
```

Run experiment on 20% of the full dataset (~1.5 hrs).
```
python classifier.py -d civilcomments -f 0.2 -s 0.1 -b 1000 -o results/civilcommets/20_percent -n 5
```

Run experiment on 100% of the full dataset (6-7 hrs).
```
python classifier.py -d civilcomments -f 1 -s 0.1 -b 10000 -o results/civilcommets/100_percent -n 5
```

### Amazon
Run experiment on 0.1% of the full dataset (3 minutes).
```
python classifier.py -d amazon -f 0.001 -s 0.1 -o results/amazon/1_10_percent -n 5
```

Run experiment on 1% of the full dataset (~1 hour).
```
python classifier.py -d amazon -f 0.1 -s 0.1 -b 100 -o results/amazon/1_percent -n 5
```

Run experiment on 10% of the full dataset (? hrs).
```
python classifier.py -d amazon -f 1 -s 0.1 -b 10000 -o results/amazon/100_percent -n 5
```