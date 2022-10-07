import matplotlib.pyplot as plt

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict
file = r'C:/Users/lapto/Desktop/machine learning/cifar-10-batches-py/data_batch_1'
data_batch_1 = unpickle(file)
print(type(data_batch_1))
print(data_batch_1.keys())


for item in data_batch_1:
    print(item, type(data_batch_1[item]))
    print("Labels:", set(data_batch_1['labels']))
X_train = data_batch_1['data']
X_train

X_train.shape

meta_file = r'C:/Users/lapto/Desktop/machine learning/cifar-10-batches-py/batches.meta'
meta_data = unpickle(meta_file)
print(type(meta_data))
print(meta_data.keys()) 
print("Label Names:", meta_data['label_names'] )

image = data_batch_1['data'][0]
image = image.reshape(3,32,32)
print(image.shape)

image = image.transpose(1,2,0)
print(image.shape)

# label names
label_name = meta_data['label_names']
# take first image
image = data_batch_1['data'][0]
# take first image label index
label = data_batch_1['labels'][0]
# Reshape the image
image = image.reshape(3,32,32)
# Transpose the image
image = image.transpose(1,2,0)
# Display the image
plt.imshow(image)
plt.title(label_name[label])