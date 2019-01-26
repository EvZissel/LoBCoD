# LoBCoD
A Local Block Coordinate Descent Algorithm for the CSC Model

This is the Matlab package that implements the LoBCoD algorithm, presented in [arXiv:1811.00312](https://arxiv.org/abs/1811.00312)
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
This pachage contains the following main modules:

| Module                    | Description 
|---------------------------|---
| LoBCoD.m                  | The main function that implements the batch LoBCoD algorithm 
| Demo.m                    | A demo script that applies the function `LoBCoD.m` on the _Fruit_ dataset 
| LoBCoD_online.m           | A function that implements the online LoBCoD algorithm 
| Demo_online.m             | A demo script that applies `LoBCoD_online.m` on a subset of _mirflickr_ dataset 
| inpainting_LoBCoD.m       | A function that implements inpainting using `LoBCoD.m`
| Demo_inpainting.m         | A demo script that applies `inpainting_LoBCoD.m`  
| Demo_Multi_Focus_Fusion.m | A demo script for implementing multi-focus image fusion 

## Examples

The training curves of the LoBCoD algorithm using `Demo.m` (trained on the _Fruit_ dataset):
<p align="center">
  <img src="./batch_training_set.png">
</p>

The converging objective value of the test set using `Demo_online.m` (trained on a subset of _mirflickr_ dataset): 
<p align="center">
  <img width="520" height="380" src="./Online_test_set.png">
</p>

Example of inpainting of the corrupted _Barbara_ image using `Demo_inpainting.m`:
<p align="center">
  <img src="./inpainting.png">
</p>

Example of multi-focus image fusion of the images _Bird_ (background and foreground in-focus) using `Demo_Multi_Focus_Fusion.m`:
<p align="center">
  <img src="./bird.png">
</p>

