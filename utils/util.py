import cv2
import numpy as np

def resize_image(path,size):
    img=cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,size)
    return img
def gen_graph(res2,limit = 100, mul = 1):

      list_label = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
      res2_index = np.argsort(res2)
      res2_sorted = np.array(res2)[np.array(res2_index)]
      res2_sorted_rev = res2_sorted[::-1];
      label_sorted = np.array(list_label)[np.array(res2_index)]
      label_sorted_rev = label_sorted[::-1]
      res2_sorted_beautify = np.around(res2_sorted * 100, 2)

      limit = np.clip(limit, 1, 100)

      limit_ratio = 100 / limit
      font_size = int(12 * mul * limit_ratio)
      bar_width = .6
      figure_size = (int(120 * mul), int(15 * mul * np.clip(limit_ratio, 1, 10)))
      height_deviation = .002 * (mul / 2)
      line_size = (mul * limit_ratio) / 2

      print("Generating graph, this could take a moment...")

      plt.rc('font', size=font_size)
      fig = plt.figure(figsize=figure_size)
      ax = fig.add_subplot()
      bars = ax.bar(label_sorted_rev[:limit], res2_sorted_rev[:limit], width=bar_width)
      ax.set_title('Prediction for Model', size=font_size * 2.5, weight='bold')
      for i, v in enumerate(res2_sorted_rev[:limit]):
          ax.text(i, v + height_deviation, str(np.around(v * 100, 2)) + "%", color='blue', fontweight='bold', ha = 'center')

      plt.grid(axis='y', color='gray', linestyle='--', linewidth=line_size, alpha=.6)
      plt.show()
