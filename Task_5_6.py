import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import match_template
from skimage import io

#Task 5

image = io.imread('coins.jpg')
coin = image[170:220, 75:130]
result = match_template(image, coin)
ij = np.unravel_index(np.argmax(result), result.shape)
x, y = ij[::-1]
fig = plt.figure(figsize=(8, 3))
ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2)
ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2)
ax1.imshow(coin, cmap=plt.cm.gray)
ax1.set_axis_off()
ax1.set_title('template')
ax2.imshow(image, cmap=plt.cm.gray)
ax2.set_axis_off()
ax2.set_title('image')
# highlight matched region
hcoin, wcoin = coin.shape
rect = plt.Rectangle((x, y), wcoin, hcoin, edgecolor='r', facecolor='none')
ax2.add_patch(rect)
ax3.imshow(result)
ax3.set_axis_off()
ax3.set_title('`match_template`\nresult')
# highlight matched region
ax3.autoscale(False)
ax3.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)
plt.show()

#Task 6

# Own implementation of template matching algorithm using Sum of Squared Differences (SSD)
def template_matching_ssd(image, template):
    # Get the dimensions of the image and the template
    image_h, image_w = image.shape
    template_h, template_w = template.shape

    # Initialize an empty result array to store SSD values
    result = np.zeros((image_h - template_h + 1, image_w - template_w + 1))

    # Slide the template across the image
    for i in range(result.shape[0]):  # Height of the image region
        for j in range(result.shape[1]):  # Width of the image region
            # Extract the current region of the image
            image_region = image[i:i + template_h, j:j + template_w]

            # Compute the sum of squared differences (SSD) between the template and the region
            ssd = np.sum((image_region - template) ** 2)

            # Store the SSD in the result array
            result[i, j] = ssd

    # Since we're using SSD, the best match will be the minimum value
    return result


# Load the main image and the template
# Already loaded the image in task 5

# Let's take a small portion of the image as the template
# This will be a coin from the image
# Using coordinates to extract the template (a coin from task 5)
template = coin

# Step 4: Compute the result using the manual template matching function
result = template_matching_ssd(image, template)

# Step 5: Find the location of the best match (minimum SSD)
ij = np.unravel_index(np.argmin(result), result.shape)  # Get the indices of the minimum SSD value
x, y = ij[::-1]  # Reverse to get x, y coordinates

# Step 6: Plot the result
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
# highlight matched region
hcoin, wcoin = template.shape
rect = plt.Rectangle((x, y), wcoin, hcoin, edgecolor='r', facecolor='none')
ax3.add_patch(rect)
ax3.axis('off')

plt.show()
