# Semi-Global Matching

Fork of the GitHub project "Semi-Global Matching" by beaupreda.

Implementation of the Semi-Global Matching algorithm in Python.
This version also includes usage of both BRIEF and HOG descriptors and also the usage of
the BMPRE (Bad Matching Pixels Relative Error) metric.

![](figures/cones.png)

![](figures/teddy.png)

#### Dependencies
* [OpenCV](https://opencv.org/)
* [NumPy](https://numpy.org/)

#### Instructions
```
$ git clone https://github.com/raphaelroyerrivard/semi-global-matching.git
$ cd path/to/semi-global-matching
```

#### Usage
```
python3 sgm.py --input [DIRECTORY TREE] --disp [MAXIMUM DISPARITY] --images [TRUE OR FALSE] --eval [TRUE OR FALSE] --descriptor [BRIEF OR HOG OR census]
```

#### Example
```
python3 sgm.py --input ./datasets --disp 64 --images True --eval True --descriptor HOG
```

#### Other implementations
* [C++](https://github.com/epiception/SGM-Census)
* [MATLAB](https://github.com/kobybibas/SemiGlobalMathingImplementation)
* [CUDA](https://github.com/fixstars/libSGM)

#### References
* [Stereo Processing by Semi-Global Matching and Mutual Information](https://core.ac.uk/download/pdf/11134866.pdf)
* [LUNOKHOD SGM Blog Post](http://lunokhod.org/?p=1356)