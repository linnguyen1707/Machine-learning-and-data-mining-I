import pandas as pd
import glob
import os


first_500_epoch = pd.read_csv("../history-report/CNN-history.csv")
#next_200_epoch = pd.read_csv("./cnn_with_dropout_batchnorm_300ep/history/cnn3model.csv")
next_750_epoch = pd.read_csv("./cnn_with_dropout_batchnorm_750ep/history/cnn3model.csv")

file = [first_500_epoch,next_750_epoch]
#print(first_100_epoch)
history = pd.concat(file,ignore_index=True)
history['epoch']=[x for x in range(0,750)]
print(history.index)
#history.to_csv("./test1.csv",index=False)
print(history['epoch'])
history.to_csv("../history-report/CNN-history-v2.csv",index=False)
