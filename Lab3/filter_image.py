from PIL import Image, ImageOps;
import numpy as np;
import sys;

########################################
# Validate Input / Load image / Setup
########################################

if len(sys.argv) != 3:
    # TODO maybe add filter selection from command line later 
    print("Usage: python filter_image.py <image_path> <filter_square_size>");
    sys.exit();

image_path = sys.argv[1];
filter_size = sys.argv[2]; 

if not filter_size.isdigit():
    print(f"<filter_square_size> is not a valid integer: {filter_size}");
    sys.exit();
else:
    filter_size = int(filter_size);

try:
    image = Image.open(image_path);
except Exception as e:
    print(f"Error opening image: {e}");
    sys.exit();

########################################
# Calculate output matrix and 
# important matrix values
########################################

def filter_channel(channel, filter_matrix):
    channel_array = np.array(channel);

    output_height = channel_array.shape[0] - (filter_matrix.shape[0] - 1);
    output_width = channel_array.shape[1] - (filter_matrix.shape[1] - 1);
    output_array = np.full((output_height, output_width), None, dtype=object)

    for y, x in np.ndindex(output_array.shape):
        channel_slice = channel_array[y: y + filter_matrix.shape[0], 
                                      x: x + filter_matrix.shape[1]];
        output_array[y, x] = np.sum((channel_slice * filter_matrix));

    return Image.fromarray(output_array.astype(np.uint8), mode='L');

def filter_image(image, filter_matrix):
    channels = image.split(); 
    filtered_channels = [filter_channel(channel, filter_matrix) for channel in channels]
    return Image.merge(image.mode, filtered_channels);

########################################
# Simple average 
########################################

def create_filter(size, fn):
    array = np.array([[fn(x, y) for x in range(size)] for y in range(size)]);
    return array / np.sum(array);

########################################
# Testing code
########################################

diagonal = lambda x, y: x - y == 0 or x - y == filter_size - 1;
d_filter = create_filter(filter_size, diagonal);

filtered_image = filter_image(image, d_filter); 
filtered_image.show();

