import os
import shutil
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from customdata import CustomData # internal
import models # internal

# argparser
parser = argparse.ArgumentParser(description='training')

parser.add_argument("--traindatadir", type=str, metavar="PATH", required=True, help="directory to train datasets")
parser.add_argument("--valdatadir", type=str, metavar="PATH", required=True, help="directory to val datasets")
parser.add_argument("--modelname", type=str, metavar="STRING", required=True, help="type of models based on input format: default, no_footprint, no_strand, no_size, no_cleavage_profile")
parser.add_argument("--batchsize", type=int, metavar="N", required=True, help="batchsize for training dataset")
parser.add_argument("--epochnumber", type=int, metavar="N", required=True, help="training epoch number")
parser.add_argument("--learnrate", type=float, metavar="N", required=True, help="learning rate")
parser.add_argument("--trainrecorddir", type=str, metavar="PATH", required=True, help="output directory for traning record")
parser.add_argument("--bestmodeldir", type=str, metavar="PATH", required=True, help="output directory for best models")
parser.add_argument("--prefix", type=str, metavar="STRING", required=True, help="output file name prefix")

args = parser.parse_args()

traindatadir = args.traindatadir
valdatadir = args.valdatadir
modelname = args.modelname
batchsize = args.batchsize
epochnumber = args.epochnumber
learnrate = args.learnrate
trainrecorddir = args.trainrecorddir
bestmodeldir = args.bestmodeldir
prefix=args.prefix

# Define model
if modelname == "default":
    model = models.default()
elif modelname == "no_footprint":
    model = models.no_footprint()
elif modelname == "no_strand" or modelname == "no_size":
    model = models.no_strand_or_no_size()
elif modelname == "no_cleavage_profile":
    model = models.no_cleavage_profile()

# Define batch size for training and validation datasets
traindatanumber=len([filename for filename in os.listdir(traindatadir) if os.path.isfile(os.path.join(traindatadir, filename))])
valdatanumber=len([filename for filename in os.listdir(valdatadir) if os.path.isfile(os.path.join(valdatadir, filename))])

if int(traindatanumber//batchsize) > 100:
    iternumber = 100
else:
    iternumber = int(traindatanumber//batchsize)

# Define Loss/criterion and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learnrate) # Adam

# Training
create_train_record = open("".join([trainrecorddir, "/", prefix, "_train_record.txt"]),"w")
create_train_record.close()

train = CustomData(datadir=traindatadir)
val = CustomData(datadir=valdatadir)

val_loss_1 = 1
val_loss_2 = 1
val_loss_3 = 1

for epoch in range(1, epochnumber+1):

    train_loader = torch.utils.data.DataLoader(train, batch_size=batchsize, shuffle=True)
    train_data_iter = iter(train_loader)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batchsize//7, shuffle=True)
    data_val_iter = iter(val_loader)

    for i in range(1, iternumber+1):

        # zero gradient
        optimizer.zero_grad()

        # train dataset
        train_inputs, train_labels, train_scores = train_data_iter.next()
        x = train_inputs
        y = model(x)
        train_loss = criterion(y, train_labels)

        # val dataset
        val_inputs, val_labels, val_scores = data_val_iter.next()
        x = val_inputs
        y = model(x)
        val_loss = criterion(y, val_labels)

        # update weights
        train_loss.backward()
        optimizer.step()

        # record
        loss_record = "\t".join([str(epoch), str(i), str(train_loss.item()), str(val_loss.item()), "\n"])
        write_train_record = open("".join([trainrecorddir, "/", prefix, "_train_record.txt"]),"a")
        write_train_record.write(loss_record)
        write_train_record.close()

        # save the trained model
        if val_loss.item() < val_loss_1:
            if os.path.isfile("".join([bestmodeldir, "/", prefix, "_bestmodel_1.pt"])):
                os.remove("".join([bestmodeldir, "/", prefix, "_bestmodel_1.pt"]))
            torch.save(model.state_dict(), "".join([bestmodeldir, "/", prefix, "_bestmodel_1.pt"]))
            val_loss_1 = val_loss.item()
        elif val_loss.item() < val_loss_2:
            if os.path.isfile("".join([bestmodeldir, "/", prefix, "_bestmodel_2.pt"])):
                os.remove("".join([bestmodeldir, "/", prefix, "_bestmodel_2.pt"]))
            torch.save(model.state_dict(), "".join([bestmodeldir, "/", prefix, "_bestmodel_2.pt"]))
            val_loss_2 = val_loss.item()
        elif val_loss.item() < val_loss_3:
            if os.path.isfile("".join([bestmodeldir, "/", prefix, "_bestmodel_3.pt"])):
                os.remove("".join([bestmodeldir, "/", prefix, "_bestmodel_3.pt"]))
            torch.save(model.state_dict(), "".join([bestmodeldir, "/", prefix, "_bestmodel_3.pt"]))
            val_loss_3 = val_loss.item()
