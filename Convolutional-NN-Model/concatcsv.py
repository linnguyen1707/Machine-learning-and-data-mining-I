import pandas as pd
import glob
import os


first_100_epoch = pd.read_csv("./cnn_with_dropout_batchnorm_first100ep/history/cnn3model.csv")
#next_200_epoch = pd.read_csv("./cnn_with_dropout_batchnorm_300ep/history/cnn3model.csv")
last_200_epoch = pd.read_csv("./cnn_with_dropout_batchnorm_500ep/history/cnn3model.csv")
file = [first_100_epoch,last_200_epoch]
#print(first_100_epoch)
history = pd.concat(file,ignore_index=True)
print(history.index)
#history.to_csv("./test1.csv")
history['epoch']=[x for x in range(0,500)]
print(history['epoch'])
history.to_csv("../history-report/CNN-history.csv",index=False)
