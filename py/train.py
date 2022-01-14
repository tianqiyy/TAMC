import os
import shutil
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from customdata import CustomData # internal
import models # internal

# argparser
parser = argparse.ArgumentParser(description='training')

parser.add_argument("--traindatadir", type=str, metavar="PATH", required=True, help="directory to training datasets")
parser.add_argument("--valdatadir", type=str, metavar="PATH", required=True, help="directory to validating datasets")
parser.add_argument("--modelname", type=str, metavar="STRING", required=True, help="type of models based on input format: default, no_footprint, no_strand, no_size, no_cleavage_profile")
parser.add_argument("--batchsize", type=int, metavar="N", required=True, help="args.batchsize for training dataset")
parser.add_argument("--epochnumber", type=int, metavar="N", required=True, help="training epoch number")
parser.add_argument("--learnrate", type=float, metavar="N", required=True, help="learning rate")
parser.add_argument("--trainrecorddir", type=str, metavar="PATH", required=True, help="output directory for traning record")
parser.add_argument("--bestmodeldir", type=str, metavar="PATH", required=True, help="output directory for best models")
parser.add_argument("--prefix", type=str, metavar="STRING", required=True, help="output file name args.prefix")

args = parser.parse_args()

# Define model
if args.modelname == "default":
    model = models.default()

# Define iteration number:
traindatanumber=len([filename for filename in os.listdir(args.traindatadir) if os.path.isfile(os.path.join(args.traindatadir, filename))])
valdatanumber=len([filename for filename in os.listdir(args.valdatadir) if os.path.isfile(os.path.join(args.valdatadir, filename))])

if int(traindatanumber//args.batchsize) > 100:
    iternumber = 100
else:
    iternumber = int(traindatanumber//args.batchsize)

# Define Loss criterion and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learnrate) # Adam

# Training
create_train_record = open("".join([args.trainrecorddir, "/", args.prefix, "_train_record.txt"]),"w")
create_train_record.close()

train = CustomData(datadir=args.traindatadir)
val = CustomData(datadir=args.valdatadir)

val_loss_1 = 1
val_loss_2 = 1
val_loss_3 = 1

for epoch in range(1, args.epochnumber+1):

    train_loader = torch.utils.data.DataLoader(train, batch_size=args.batchsize, shuffle=True)
    train_data_iter = iter(train_loader)
    val_loader = torch.utils.data.DataLoader(val, batch_size=args.batchsize//7, shuffle=True)
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
        write_train_record = open("".join([args.trainrecorddir, "/", args.prefix, "_train_record.txt"]),"a")
        write_train_record.write(loss_record)
        write_train_record.close()

        # save the trained model
        if val_loss.item() < val_loss_1:
            if os.path.isfile("".join([args.bestmodeldir, "/", args.prefix, "_bestmodel_1.pt"])):
                os.remove("".join([args.bestmodeldir, "/", args.prefix, "_bestmodel_1.pt"]))
            torch.save(model.state_dict(), "".join([args.bestmodeldir, "/", args.prefix, "_bestmodel_1.pt"]))
            val_loss_1 = val_loss.item()
        elif val_loss.item() < val_loss_2:
            if os.path.isfile("".join([args.bestmodeldir, "/", args.prefix, "_bestmodel_2.pt"])):
                os.remove("".join([args.bestmodeldir, "/", args.prefix, "_bestmodel_2.pt"]))
            torch.save(model.state_dict(), "".join([args.bestmodeldir, "/", args.prefix, "_bestmodel_2.pt"]))
            val_loss_2 = val_loss.item()
        elif val_loss.item() < val_loss_3:
            if os.path.isfile("".join([args.bestmodeldir, "/", args.prefix, "_bestmodel_3.pt"])):
                os.remove("".join([args.bestmodeldir, "/", args.prefix, "_bestmodel_3.pt"]))
            torch.save(model.state_dict(), "".join([args.bestmodeldir, "/", args.prefix, "_bestmodel_3.pt"]))
            val_loss_3 = val_loss.item()
