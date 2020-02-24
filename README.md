# Semi-Global Matching

Implementation of the Semi-Global Matching algorithm in Python.

![](figures/cones.png)

![](figures/teddy.png)

#### Dependencies
* [OpenCV](https://opencv.org/)
* [NumPy](https://numpy.org/)

#### Instructions
```
$ git clone https://github.com/beaupreda/semi-global-matching.git
$ cd path/to/semi-global-matching
```

#### Usage
```
python3 sgm.py --left [LEFT IMAGE NAME] --right [RIGHT IMAGE NAME] --left_gt [LEFT GT IMAGE NAME] --right_gt [RIGHT GT IMAGE NAME] --output [OUTPUT FOLDER] --disp [MAXIMUM DISPARITY] --images [TRUE OR FALSE] --eval [TRUE OR FALSE] --descriptor [BRIEF OR HOG] --orientations [ORIENTATION COUNT]
```

#### Example
```
python3 sgm.py --left cones/im2.png --right cones/im6.png --left_gt cones/disp2.png --right_gt cones/disp6.png --output cones_HOG --disp 64 --images True --eval True --descriptor HOG --orientations 9
```

#### Other implementations
* [C++](https://github.com/epiception/SGM-Census)
* [MATLAB](https://github.com/kobybibas/SemiGlobalMathingImplementation)
* [CUDA](https://github.com/fixstars/libSGM)

#### References
* [Stereo Processing by Semi-Global Matching and Mutual Information](https://core.ac.uk/download/pdf/11134866.pdf)
* [LUNOKHOD SGM Blog Post](http://lunokhod.org/?p=1356)