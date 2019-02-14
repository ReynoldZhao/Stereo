# Stereo

## Introduction

* [Photometric Stereo](http://www.cs.cornell.edu/courses/cs5670/2018sp/projects/pa4/index.html#photometricstereo) Given a stack of images taken from the same viewpoint under different, known illumination directions, to recover the albedo and normals of the object surface.
* [Plane sweep Stereo](http://www.cs.cornell.edu/courses/cs5670/2018sp/projects/pa4/index.html#planesweepstereo) Given two calibrated images of the same scene, but taken from different viewpoints,  to recover a rough depth map.
* [Depth map reconstruction](http://www.cs.cornell.edu/courses/cs5670/2018sp/projects/pa4/index.html#depthmapreconstruction) Given a normal map, depth map, or both, reconstruct a 3D mesh.

Click [here](http://www.cs.cornell.edu/courses/cs5670/2018sp/projects/pa4/index.html) to view projects introduction. 

## Features

* Implement photometric stereo method, with optical vision knowledge and known illumination directions, recover albedo and normals of the object surface.

* Implement plane sweep stereo, recover a rough depth map

* Resconstruct 3D mesh of objects

  


## Structure

| Name           | Function                                                     |
| -------------- | ------------------------------------------------------------ |
| resources/     | available images to create  panorama                         |
| src/tests.py   | test function in student.py whether functions results are correct as origin pre-setting parameters |
| src/student.py | Recover an object’s stereogram and depth map with it’s images from different angels. Using algorithms of Photometric Stereo and Plane-sweep Stereo |

## Usages

### Requirements

* Linux / Windows / MacOS
* python 2.7 / python 3.5
* cv2
* numpy
* pandas
* scipy
* To view the output result Intuitively, you will need ImageMagick, MeshLab and nose.

### Compilation

``` python
cd data 

sh download.sh 

cd .. 

mkdir output 

mkdir temp

#Part1:You will need ImageMagick, MeshLab and nose. If you are using the class VM then run:
sudo apt-get install imagemagick meshlab python-nose

#Given a stack of images taken from the same viewpoint under different, known illumination directions, your task is to recover the albedo and normals of the object surface. E.g.,

python photometric_stereo.py tentacle

#Part2: Given two calibrated images of the same scene, but taken from different viewpoints, your task is to recover a rough depth map. For example, if you use the tentacle dataset

python plane_sweep_stereo.py tentacle

#Part3: Given a normal map, depth map, or both, reconstruct a mesh. e.g.,

python combine.py tentacle both 
or 
./combine tentacle both 

```

## Examples

### Albedo and normals of the object surface, depth map and 3D mesh

| ![](https://github.com/ReynoldZhao/Project4_Stereo/raw/master/Project_4Results/Flowers_projected.gif) | ![](https://github.com/ReynoldZhao/Project4_Stereo/raw/master/Project_4Results/Flowers_ncc.gif) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![](https://github.com/ReynoldZhao/Project4_Stereo/raw/master/Project_4Results/tentacle_ncc.gif) | ![](https://github.com/ReynoldZhao/Project4_Stereo/raw/master/Project_4Results/tentacle_projected.gif) |
