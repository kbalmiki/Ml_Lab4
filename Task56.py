import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import match_template
from skimage import data, color, feature
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
template = image[170:220, 75:130]  # Coordinates to extract the template (a coin)

# Perform template matching using match_template
result = feature.match_template(image, template)

# Find the location of the best match
ij = np.unravel_index(np.argmax(result), result.shape)  # Get the indices of the maximum match
x, y = ij[::-1]  # Reverse to get x, y coordinates

#  Plot the result
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
