import matplotlib.pyplot as plt

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict
file = r'C:/Users/lapto/Desktop/machine learning/cifar-100-python/train'
train_data = unpickle(file)
print(type(train_data))
print(train_data.keys())


for item in train_data:
    print(item, type(train_data[item]))
    print("Fine Labels:", set(train_data['fine_labels']))
    print("Coarse Labels:", set(train_data['coarse_labels']))
X_train = train_data['data']
X_train

X_train.shape

meta_file = r'C:/Users/lapto/Desktop/machine learning/cifar-100-python/meta'
meta_data = unpickle(meta_file)
print(type(meta_data))
print(meta_data.keys()) 
print("Fine Label Names:", meta_data['fine_label_names'] )
print("Coarse Label Names:", meta_data['coarse_label_names'] )

#We reshape and transpose dataset as we did in the CIFAR-10.
X_train = train_data['data']
# Reshape the whole image data
X_train = X_train.reshape(len(X_train),3,32,32)
# Transpose the whole data
X_train = X_train.transpose(0,2,3,1)

# Python 3 program to visualize 4th image
import matplotlib.pyplot as plt
# take 4th image from training data
image = train_data['data'][3]
# reshape and transpose the image
image = image.reshape(3,32,32).transpose(1,2,0)
# take coarse and fine labels of the image 
c_label = train_data['coarse_labels'][3]
f_label = train_data['fine_labels'][3]
# take coarse and fine label names of the image
coarse_name = meta_data['coarse_label_names'][c_label]
fine_name = meta_data['fine_label_names'][f_label]
# dispaly the image
plt.imshow(image)
plt.title("Coarse Label Name:{} \n Fine Label Name:{}"
          .format(coarse_name, fine_name))

# Python 3 program to visualize 4th image
import matplotlib.pyplot as plt
import numpy as np
# take the images data from training data
images = train_data['data']
# reshape and transpose the images
images = images.reshape(len(images),3,32,32).transpose(0,2,3,1)
# take coarse and fine labels of the images 
c_labels = train_data['coarse_labels']
# print(c_labels)
f_labels = train_data['fine_labels']
# take coarse and fine label names of the images
coarse_names = meta_data['coarse_label_names']
fine_names = meta_data['fine_label_names']


# dispaly random nine images
# define row and column of figure
rows, columns = 3, 3
# take random image idex id
imageId = np.random.randint(0, len(images), rows * columns)
# take images for above random image ids
images = images[imageId]
# take coarse and fine labels for these images only
c_labels = [c_labels[i] for i in imageId]
f_labels = [f_labels[j] for j in imageId]
# define figure
fig=plt.figure(figsize=(8, 10))
# visualize these random images
for i in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, i)
    plt.imshow(images[i-1])
    plt.xticks([])
    plt.yticks([])
    plt.title("{} \n {}"
          .format(coarse_names[c_labels[i-1]], fine_names[f_labels[i-1]]))
plt.show()
