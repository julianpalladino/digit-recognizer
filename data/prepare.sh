#!/bin/bash
for file in train test; do
    gunzip -k $file.csv.gz
done
