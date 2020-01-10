# indoor-scene-layout-labeling
LSUN dataset indoor scene layout labeling via transfer learning with segmentation ResNet101

## Prerequisites

- cv2
- PyTorch
- Torchvision

## Usage 
The root directory should be namded "Bounding_box/"

- To label image using trained model:
```
$python main.py "filename"
```
- To train a new model:
```
$python learn.py "filename"
```
alternatively, utilize cloud GPUs with "BBox.ipynb". This requires uploading LSUN data as well.

Trained models should be saved in the root directory

Modify training paramters in "params.py"

## Data
Download LSUN room dataset at [LSUN](https://drive.google.com/file/d/1e40AC_9CwgWPQL9eh18y2k9u4O0X3rl4/view) and place under "data/" directory

Credit: [liamw96](https://github.com/liamw96/pytorch.room.layout)

## Example

Input: 
![alt text](https://github.com/hyc96/Indoor-scene-vanish-point-detection-and-line-labeling/blob/master/input/1.jpg)
Output:
![alt text](https://github.com/hyc96/Indoor-scene-vanish-point-detection-and-line-labeling/blob/master/output/membership_1.jpg)

Vanishing lines are labeled with three directions, the thrid label indicates irrelavent lines.

## License

see the [LICENSE.md](LICENSE.md) file for details

## References 
This project is generally implemented based on:
1. Mallya, A. & Lazebnik, S. (2015). Learning Informative Edge Maps for Indoor Scene Layout Prediction
2. Schwing, A. G. & Urtasun, R. (2012). Efficient Exact Inference for 3D Indoor Scene Understanding
3. Hedau, V., Hoiem, D. & Forsyth, D. A. (2009). Recovering the spatial layout of cluttered rooms
4. Rother, C. (2000). A New Approach for Vanishing Point Detection in Architectural Environments
5. Tardif, J.-P. (2009). Non-iterative approach for fast and accurate vanishing point detection
6. Denis, P., Elder, J. H. & Estrada, F. J. (2008). Efficient Edge-Based Methods for Estimating Manhattan Frames in Urban Imagery

Contact me for more detailed report on the implementation.

## Contact
huaiyuc@seas.upenn.edu
