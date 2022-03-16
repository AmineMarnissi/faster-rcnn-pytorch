import os
net = "res18"
dataset = "kaist_tr"
start_epoch = 2
max_epochs = 10
s = 1

for i in range(start_epoch, max_epochs + 1):
    model_dir = "./models/{}/{}/faster_rcnn_{}_{}_7571.pth".format(net,"kaist",s,i)
    command = "python test_net.py --dataset {} --net {}  --load_name {} ".format(dataset,net,model_dir)
    os.system(command)
