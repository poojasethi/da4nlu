# Domain Adaptation for NLU
Comparing self-learning vs. active learning for NLU domain adaptation

## Civil Comments
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

## Amazon
Run experiment on 0.1% of the full dataset (3 minutes).
```
python classifier.py -d amazon -f 0.001 -s 0.1 -o results/amazon/1_10_percent -n 5
```

Run experiment on 1% of the full dataset (~1 hour).
```
python classifier.py -d amazon -f 0.1 -s 0.1 -b 100 -o results/amazon/1_percent -n 5
```

Run experiment on 100% of the full dataset (6-7 hrs).
```
python classifier.py -d amazon -f 1 -s 0.1 -b 10000 -o results/amazon/100_percent -n 5
```