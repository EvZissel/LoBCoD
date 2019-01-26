# LoBCoD
A Local Block Coordinate Descent Algorithm for the CSC Model

This is the Matlab Package for the LoBCoD algorithm, available on [arXiv:1811.00312](https://arxiv.org/abs/1811.00312)
```
E. Zisselman, J. Sulam and M. Elad, "A Local Block Coordinate Descent Algorithm
for the Convolutional Sparse Coding Model". 
``` 

## Installation

Download the package to a local folder (e.g. ~/LoBCoD/) by running: 
```console
git clone https://github.com/EvZissel/LoBCoD.git
```

Open Matlab and navigate to the folder (~/LoBCoD/).

## Dependencies

This code uses the following packages: 
* [vlfeat](https://github.com/vlfeat/vlfeat) - An open library of computer vision algorithms.
* [SPAMS optimization toolbox](http://spams-devel.gforge.inria.fr/) - For its implementation of the batch LARS algorithm.

For Windows
```
This code is self-contained and includes all the precompiled packages.
```

## Description
This pachage contains the following modules:

| Module                    | Description 
|---------------------------|---
| LoBCoD.m                  | The main function that implements the batch LoBCoD algorithm 
| Demo.m                    | A demo script demonstrates the batch LoBCoD on the `Fruit` dataset 
| LoBCoD_online.m           | A function that implements the online LoBCoD algorithm 
| Demo_online.m             | A demo script demonstrates the online LoBCoD on a subset of `mirflickr` dataset 
| inpainting_LoBCoD.m       | A function that implements the inpainting application 
| Demo_inpainting.m         | A demo script demonstrates implementation for inpainting 
| Demo_Multi_Focus_Fusion.m | A demo script for implementation of multi-focus image fusion 

## Examples
![Figure 1](./batch_training_set.png)

<p align="center">
  <img width="460" height="300" src="./Online_test_set.png">
</p>

![Figure 3](./inpainting.png)

<p align="center">
  <img src="./bird.png">
</p>

