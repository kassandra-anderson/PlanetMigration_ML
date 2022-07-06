#!/bin/bash

rm alldata.csv

head -1 batch_seed1.csv > alldata.csv
#for fname in *.csv
for i in {1..5}
do
    fname="batch_seed"$i".csv"
    echo $fname
    tail -n+2 $fname >> alldata.csv
done
