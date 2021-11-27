# da4nlu
Comparing self-learning vs. active learning for NLU domain adaptation

Run experiment on 0.1% of the full dataset (< 10 min).
```
python classifier.py -d civilcomments -f 0.001 -s 0.1 -o results_1/10_percent -n 5
```

Run experiment on 1% of the full dataset (~30 min).
```
python classifier.py -d civilcomments -f 0.1 -s 0.1 -b 100 -o results_1_percent -n 5
```

Run experiment on 20% of the full dataset (~1.5 hrs).
```
python classifier.py -d civilcomments -f 0.2 -s 0.1 -b 1000 -o results_20_percent -n 5
```

Run experiment on 100% of the full dataset.
```
python classifier.py -d civilcomments -f 1 -s 0.1 -b 10000 -o results_100_percent -n 5
```