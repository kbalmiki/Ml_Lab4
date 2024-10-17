import os
from skimage.color import rgb2gray
from skimage import io
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize

image = io.imread('coins.jpg')
astronaut_image = io.imread('astronaut.jpg')


print(image.shape)
print(image.dtype)
print(image[1,100])

print(astronaut_image.shape)
print(astronaut_image.dtype)
print(astronaut_image[1,100])

io.imshow(image)
io.imshow(astronaut_image)

io.show()

ax = plt.imshow(image)
plt.show()

#Task 2

def show_grayscale_image(image):
    grayscale = None
    # If the image has 2 dimensions, it's already grayscale
    if image.ndim == 2:
        grayscale = image
    else:
        # Otherwise, convert it to grayscale
        grayscale = rgb2gray(image)
    return grayscale


coin_grayscale = show_grayscale_image(image)
astronaut_grayscale = show_grayscale_image(astronaut_image)

print("Grayscale image shape for coin:", coin_grayscale)
print("Grayscale image shape for astronaut:", astronaut_grayscale)


io.imshow(image)
io.show()

# Display the grayscale image
plt.imshow(coin_grayscale, cmap='gray')
plt.imshow(astronaut_image, cmap='gray')



plt.show()

print('after grayscale shape coin',image.shape)
print('after grayscale dtype coin',image.dtype)

#Task 3
image_rescaled = rescale(coin_grayscale, 0.5, anti_aliasing=False)
image_resized=resize(coin_grayscale, (image.shape[0] // 4, image.shape[1]// 4),
anti_aliasing=True)

astro_image_rescaled = rescale(coin_grayscale, 0.75, anti_aliasing=False)
astro_image_resized=resize(coin_grayscale, (image.shape[0] // 4, image.shape[1]// 4),anti_aliasing=True)