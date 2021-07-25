#Simple Linear Iterative Clustering
run slics.py to generate segmented images. My image is *my_superpixel_image.png* and the skimage one is *skimage_superpixel_image.png* 

|![image](my superpixel image.png "superpixel image")|
|---|
|*This is my image.*|

|![image](skimage superpixel image.png "superpixel image")|
|---|
|This is the skimage slic image.|

The biggest con of my implementation is that it is very slow. It uses for loops to iterate over centroids and assign labels, which does not take advantage of numpy's parallel processing.
Additional problems with it are that it does not initialize centroids to the lowest gradient positions, which could noise to interfere with the clusters obtained.
Also, I do not enforce connectivity.
However, my implementation is much shorter.

#Visual Attention in Deep Neural Networks



## Demo 1

![](Ellie%20Mae%203.jpg)

This image is unmodified.

|              Predicted class               |                       #1 tabby                        |                       #2 Egyptian_cat                        |                       #3 tiger cat                        |
| :----------------------------------------: | :---------------------------------------------------: | :----------------------------------------------------------: | :-------------------------------------------------------: |
|        Grad-CAM                            |    ![](results/demo1/unaltered/2-resnet152-gradcam-layer4-tabby.png)     |    ![](results/demo1/unaltered/2-resnet152-gradcam-layer4-Egyptian_cat.png)     |    ![](results/demo1/unaltered/2-resnet152-gradcam-layer4-tiger_cat.png)     |
|          Vanilla backpropagation           |        ![](results/demo1/unaltered/2-resnet152-vanilla-tabby.png)        |        ![](results/demo1/unaltered/2-resnet152-vanilla-Egyptian_cat.png)        |        ![](results/demo1/unaltered/2-resnet152-vanilla-tiger_cat.png)        |
|    Guided Grad-CAM                         | ![](results/demo1/unaltered/2-resnet152-guided-tabby.png) | ![](results/demo1/unaltered/2-resnet152-guided-Egyptian_cat.png) | ![](results/demo1/unaltered/2-resnet152-guided-tiger_cat.png) |

![](Ellie%20Mae%203%20altered.jpg)

This image has 3 concentric red hexagons to fool the neural network.

|              Predicted class               |                       #1 jeans                        |                       #2 mousetrap                        |                       #3 pillow                        |
| :----------------------------------------: | :---------------------------------------------------: | :----------------------------------------------------------: | :-------------------------------------------------------: |
|        Grad-CAM      |    ![](results/demo1/perturbed/2-resnet152-gradcam-layer4-jean.png)     |    ![](results/demo1/perturbed/2-resnet152-gradcam-layer4-mousetrap.png)     |    ![](results/demo1/perturbed/2-resnet152-gradcam-layer4-pillow.png)     |
|          Vanilla backpropagation           |        ![](results/demo1/perturbed/2-resnet152-vanilla-jean.png)        |        ![](results/demo1/perturbed/2-resnet152-vanilla-mousetrap.png)        |        ![](results/demo1/perturbed/2-resnet152-vanilla-pillow.png)        |
|    Guided Grad-CAM    | ![](results/demo1/perturbed/2-resnet152-guided_gradcam-layer4-jean.png) | ![](results/demo1/perturbed/2-resnet152-guided_gradcam-layer4-mousetrap.png) | ![](results/demo1/perturbed/2-resnet152-guided_gradcam-layer4-pillow.png) |



## Demo 2

Grad-CAM maps for "bull mastiff" class, at different layers of ResNet-152 (hardcoded).
This image is unmodified.

|            Layer             |                     ```relu```                      |                     ```layer1```                      |                     ```layer2```                      |                     ```layer3```                      |                     ```layer4```                      |
| :--------------------------: | :-------------------------------------------------: | :---------------------------------------------------: | :---------------------------------------------------: | :---------------------------------------------------: | :---------------------------------------------------: |
| Grad-CAM | ![](results/demo2/unaltered/2-resnet152-gradcam-relu-bull_mastiff.png) | ![](results/demo2/unaltered/2-resnet152-gradcam-layer1-bull_mastiff.png) | ![](results/demo2/unaltered/2-resnet152-gradcam-layer2-bull_mastiff.png) | ![](results/demo2/unaltered/2-resnet152-gradcam-layer3-bull_mastiff.png) | ![](results/demo2/unaltered/2-resnet152-gradcam-layer4-bull_mastiff.png) |



Grad-CAM maps for "bull mastiff" class, at different layers of ResNet-152 (hardcoded).
This image has 3 concentric red hexagons to fool the neural network

|            Layer             |                     ```relu```                      |                     ```layer1```                      |                     ```layer2```                      |                     ```layer3```                      |                     ```layer4```                      |
| :--------------------------: | :-------------------------------------------------: | :---------------------------------------------------: | :---------------------------------------------------: | :---------------------------------------------------: | :---------------------------------------------------: |
| Grad-CAM | ![](results/demo2/perturbed/2-resnet152-gradcam-relu-bull_mastiff.png) | ![](results/demo2/perturbed/2-resnet152-gradcam-layer1-bull_mastiff.png) | ![](results/demo2/perturbed/2-resnet152-gradcam-layer2-bull_mastiff.png) | ![](results/demo2/perturbed/2-resnet152-gradcam-layer3-bull_mastiff.png) | ![](results/demo2/perturbed/2-resnet152-gradcam-layer4-bull_mastiff.png) |
