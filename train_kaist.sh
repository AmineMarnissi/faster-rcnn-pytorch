#!/bin/bash
source_dataset="kaist_tr"
net="res18"
s=1

python train_net.py --dataset ${source_dataset} --net ${net} --s ${s}
