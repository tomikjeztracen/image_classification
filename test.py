import scipy
import pandas

# Cesty k datasetu
images_dir = '/Users/tomashorak/image_classification/jpg'
labels_file = '/Users/tomashorak/image_classification/imagelabels.mat'
splits_file = '/Users/tomashorak/image_classification/setid.mat'


labels = scipy.io.loadmat(labels_file)['labels'][0]

print(len(labels))


