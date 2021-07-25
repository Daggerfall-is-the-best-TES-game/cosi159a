import numpy as np
from scipy.linalg import norm
from skimage import io, color
from skimage.segmentation import slic
from math import prod, sqrt


class Superpixel():
    def __init__(self, k):
        """
        :param k: the number of superpixels
        """
        self.k = k

    def segmentation(self, image):
        """

        :param image: is an RGB image as an ndarray
        :return: the image segmented according to the SLICS algorithm
        """
        image = color.rgb2lab(image)
        image = self.assign_image_coordinates(image)
        grid_spacing, margin, dimensions = self.initialize_grid(image)
        centroids = self.initialize_centroids(image, margin, dimensions, grid_spacing)

        def D(a, b):
            """
            This is the distance measure for the clustering
            a and b are pixel vectors of shape N X 5, where N is the number of pixels in the vector
            :return:
            """
            color_dist = norm(a[..., :3] - b[..., :3], axis=-1)
            spat_dist = norm(a[..., 3:] - b[..., 3:], axis=-1)
            m = 20
            return np.sqrt(color_dist ** 2 + (spat_dist / grid_spacing) ** 2 * m ** 2)

        labels = np.full(dimensions[:2], -1)
        old_distances = np.full(dimensions[:2], np.inf)

        for _ in range(10):
            # assignment
            for k, centroid in enumerate(centroids):
                idx = self.clip_index(centroid[3:], grid_spacing, dimensions[:2])
                new_distances = D(centroid[np.newaxis, np.newaxis], image[idx])
                mask = new_distances < old_distances[idx]
                old_distances[idx][mask] = new_distances[mask]
                labels[idx][mask] = k
            # update
            for idx, _ in enumerate(centroids):
                mask = labels == idx
                centroids[idx] = np.mean(image[mask], axis=0)

        superpixel_image = np.zeros(dimensions[:2] + (3,))
        for k, centroid in enumerate(centroids):
            mask = labels == k
            superpixel_image[mask] = centroid[np.newaxis, np.newaxis, :3]
        superpixel_image = color.lab2rgb(superpixel_image)
        return (superpixel_image * 255).astype(np.uint8)

    def initialize_grid(self, image):
        """
        :param image: H X W X 5 array
        :return: the information relevant to the initial position of the centroids.
        """
        pixel_num = prod(image.shape[:-1])
        grid_spacing = int(sqrt(pixel_num / self.k))
        margin = int(grid_spacing / 2)
        dimensions = image.shape
        return grid_spacing, margin, dimensions

    def initialize_centroids(self, image, margin, dimensions, grid_spacing):
        """
        :param image: H X W X 5 ndarray representing an image
        :param margin: the distance from the edges of the image where no centroid can be initialized
        :param dimensions: the dimensions of the image
        :param grid_spacing: the distance between the centroids to be initialized
        :return: N X 5 ndarray representing the centroids
        """
        return image[margin:dimensions[0]:grid_spacing, margin:dimensions[1]:grid_spacing].reshape(-1, 5)

    def clip_index(self, coords, grid_spacing, dimensions):
        """
        the search space is supposed to be 2S x 2S around a centroid, which may go out of bounds of the image.
        So i clip the search space representing by an index to the image array to the bounds of the image.
        :param coords: the coordinates of the centroid
        :param grid_spacing: the radius of the search space
        :param dimensions: the dimensions of the image
        :return: the index of the search space clipped to the boundaries of the image
        """
        clipped = (np.clip((coord - grid_spacing, coord + grid_spacing), 0, dim).astype(int) for coord, dim in
                   zip(coords, dimensions))
        return tuple(slice(*idx) for idx in clipped)

    def assign_image_coordinates(self, image):
        """
        :param image: ndarray H x W X 3
        :return: ndarray H x W X 5, where the last two dimensions are coordinates for the pixels. The coordinates
        of the pixels are necessary to do the distance computations.

        """
        dimensions = image.shape[:2]
        coords = np.transpose(np.mgrid[:dimensions[0], :dimensions[1]], (1, 2, 0))
        return np.concatenate((image, coords), axis=-1)


if __name__ == "__main__":
    model = Superpixel(100)
    img = io.imread("brandeis.jpg")
    my_superpixel_image = model.segmentation(img)
    io.imsave("my superpixel image.png", my_superpixel_image)
    ski_superpixel_image = slic(img, enforce_connectivity=False, multichannel=True, start_label=1).astype(np.uint8)
    io.imsave("skimage superpixel image.png", ski_superpixel_image)
