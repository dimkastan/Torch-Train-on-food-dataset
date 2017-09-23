#!/bin/bash
## Script to download and extract food dataset
## download food-101 dataset
wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
# extract data
tar -xvf food-101.tar.gz
# ready!
echo "done"
