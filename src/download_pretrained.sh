#! /bin/bash

wget -nc -r --no-parent -A '*' -nH -R "index.html*" http://machinelearning2.snu.ac.kr:8000/pretrained/
