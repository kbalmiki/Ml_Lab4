# Step 1: Import necessary libraries
from skimage import data, color, feature
import matplotlib.pyplot as plt
import numpy as np

# Step 2: Load the main image and the template
# Using the 'coins' image from skimage for demonstration (it's a grayscale image)
image = data.coins()

# Let's take a small portion of the image as the template
# This will be a coin from the image
template = image[170:220, 75:130]  # Coordinates to extract the template (a coin)

# Step 3: Perform template matching using match_template
result = feature.match_template(image, template)

# Step 4: Find the location of the best match
ij = np.unravel_index(np.argmax(result), result.shape)  # Get the indices of the maximum match
x, y = ij[::-1]  # Reverse to get x, y coordinates

# Step 5: Plot the result
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))

# Original image
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Original Image')
ax1.axis('off')

# Template image
ax2.imshow(template, cmap=plt.cm.gray)
ax2.set_title('Template')
ax2.axis('off')

# Result of template matching
ax3.imshow(image, cmap=plt.cm.gray)
ax3.set_title('Template Matching Result')
# Draw a rectangle where the best match is found
h, w = template.shape
rect = plt.Rectangle((x, y), w, h, edgecolor='r', facecolor='none')
ax3.add_patch(rect)
ax3.axis('off')

plt.show()
