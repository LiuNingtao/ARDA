#!/bin/bash
# train 14369
for i in `seq 14369`
do
	python visu_skull.py $i 0
done

# val 3070
for i in `seq 3070`
do
	python visu_skull.py $i 1
done

# test 3068
for i in `seq 3068`
do
	python visu_skull.py $i 2
done