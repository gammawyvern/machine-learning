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
    image = Image.open(image_path).convert("L");
except Exception as e:
    print(f"Error opening image: {e}");
    sys.exit();

########################################
# Calculate output matrix and 
# important matrix values
########################################

def filter_image(gray_image, filter_matrix):
    image_array = np.array(gray_image);

    output_height = image_array.shape[0] - (filter_matrix.shape[0] - 1);
    output_width = image_array.shape[0] - (filter_matrix.shape[1] - 1);

    output_array = np.full((output_height, output_width), None, dtype=object)

    for y, x in np.ndindex(output_array.shape):
        image_slice = image_array[y: y + filter_matrix.shape[0], 
                                  x: x + filter_matrix.shape[1]];

        output_array[y, x] = np.sum((image_slice * filter_matrix));

    return output_array.astype(np.uint8);

########################################
# Testing code
########################################

def get_simple_filter(size):
    average_percent = 1 / (size * size)
    return np.full((size, size), average_percent); 

# filtered_matrix = filter_image(image, my_filter);
filtered_matrix = filter_image(image, get_simple_filter(filter_size));
image_smoothed = Image.fromarray(filtered_matrix, mode='L')
image_smoothed.show();

