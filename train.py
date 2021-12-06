import numpy as np
import cv2
import glob
import pathlib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split 
#from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score
import imutils
import pickle

data_dir = pathlib.Path('CROP_DATASET')
all_image_paths = list(data_dir.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]

all_labels = list(map(lambda x: x.split('\\')[-2], all_image_paths))

label_names = sorted(item.name for item in data_dir.glob('*/') if item.is_dir())
label_names

label_to_index = dict((name, index) for index, name in enumerate(label_names))
label_to_index

all_image_labels = [label_to_index[label] for label in all_labels]

#function to read image from path and preprocess
def load_and_preprocess(path):
    path = str(path)
    img = cv2.imread(path)
    img = cv2.resize(img, (300, 300), interpolation = cv2.INTER_AREA)
    img = img / 255.
    img = img.flatten()
    return img

images = np.array(list(map(load_and_preprocess, all_image_paths)))

#spit to train set, test set
x_train, x_test, y_train, y_test = train_test_split(images, all_image_labels, test_size=0.2)

# use knn model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print(accuracy_score(y_test, y_pred))

# export model to pickle file
pickle.dump(knn, open('model_knn.pkl', 'wb'))
# load model and reuse:
#loaded_model = pickle.load(open('knn_model.pkl', 'rb'))