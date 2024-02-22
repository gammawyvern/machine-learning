from PIL import Image;
import numpy as np;

# Grayscale
img = Image.open("/home/keag/Github/machine-learning/pca-lab/input-image.bmp");

# Get mean with numpy
img_matrix = np.array(img);
img_mean = np.mean(img_matrix);

# Get covariance with mean
img_norm = img_matrix - img_mean;
img_covar = np.cov(img_norm, rowvar=False)

# Determine the eigenvectors/values from the covariance 
eig_val, eig_vec = np.linalg.eigh(img_covar);
# TODO Pairs might be messed up, may need to transpose eig_vec?
# who tf knows though?
eig_pairs = list(zip(eig_val, eig_vec)); 

# Sort and keep the 15 most important eigenvectors
eig_pairs_sorted = sorted(eig_pairs, key=lambda pair: pair[0], reverse=True);
key_eig_pairs = eig_pairs_sorted[0:15];

# Calculate the variance % of 15 chosen eigenvectors
total_eigen_sum = np.sum(eig_val);
key_eigen_sum = np.sum([pair[0] for pair in key_eig_pairs]);
variance_percent = key_eigen_sum / total_eigen_sum;

# TODO I messed this up
# Either need to keep original format of linalg.eigh,
# or alter it back here for this to work
compressed_image = np.matmul(img_norm, key_eig_pairs);
lossy_compressed_image = np.matmul(compressed_image, np.transpose(key_eig_pairs)) + img_mean;

