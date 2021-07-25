from PIL import Image
from itertools import accumulate

# load image
original = Image.open("his best customer.png")
# resize image
resized = original.resize((300, 400))
# horizontal flip
flipped = resized.transpose(Image.FLIP_LEFT_RIGHT)
# convert to black and white
black_and_white = flipped.convert("L")
# concatenate the transformed images into a row
images = [original, resized, flipped, black_and_white]


def concat_image(images):
    """takes a list of images and concatenates them
    :param images - a list of images
    :returns the concatenated image"""
    im_width, im_height = sum(image.width for image in images), max(image.height for image in images)
    row_image = Image.new("RGB", (im_width, im_height))
    origin_x_coords = list(accumulate((image.width for image in images), initial=0))[:-1]
    for x_coord, image in zip(origin_x_coords, images):
        row_image.paste(image, (x_coord, 0))
    return row_image


row_image = concat_image(images)
row_image.save("concat_image.jpg")
