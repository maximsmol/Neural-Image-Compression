#!/bin/bash

set -e

git clone git@github.com:maximsmol/Neural-Image-Compression.git

sudo add-apt-repository universe
sudo apt-get update
sudo apt install -y python3-pip

cd Neural-Image-Compression
./install.sh
