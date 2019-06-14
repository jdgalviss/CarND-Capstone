# Self-driving Car - System Integration
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

### Team
[Juan David Galvis](https://github.com/jdgalviss) jdgalviss@gmail.com

[David Cardozo](https://github.com/Davidnet) david@kiwicampus.com

[Carlos Alvarez](https://github.com/charlielito) charlie@kiwicampus.com

[John Betancourt](https://github.com/Johnbetacode) john.betancourt93@gmail.com

[Andres Rengifo](https://github.com/andresR8) anferesa239@gmail.com

## Project Overview

In this project we develop a system which integrates multiple components to drive a car autonomously (drive by wire, waypoint generation, steering and throttle control and traffic light classification). This system is first implemented in simulation and then in a real car (Carla, Udacity's self driving car.)

[//]: # (Image References)

[image1]: ./result_imgs/startup.gif "result"
[image2]: ./result_imgs/classification.gif "result2"

Following modules are implemented:
#### Traffic light Classification
Using a ssd_mobilenetv1 pretrained model (with the COCO dataset), we took data provided by [Alex lechner's group](https://github.com/alex-lechner/Traffic-Light-Classification) to implement transfer learning and obtain a Deep Neural Network  model to detect and classify traffic light on images from both simulation and Carla (udacity's self driving car). We use 2 models, one for real images and one for simulation images They should be switched in the file catkin_ws/src/tl_detector/tl_detector.py line 27.
#### Waypoint Updater
The main goal of this module is to publish a set of waypoints to be followed by the car, we have reduced the number of waypoints in the lane generation to 50 to reduce the computational load. First, the algorithm searches for the closest waypoint to the car and take the next 50 waypoints from the base waypoints. Then, we checked for the stop signal and if it is present, we generate a deceleration waypoint set based on the formula '10-10 exp(-x^2/128)' generating a soft brake behavior. Finally, the final waypoint list is published using a rostopic.
#### DBW
The drive-by-wire node implements the controllers needed to move the vehicle to follow the target waypoints. The break and throttle are regulated using a classic PID controller, the reference is the linear velocity taken from the waypoint follower. The throttle control takes into account the sign of the control output to send the corresponding brake value when it is necessary.  The steering control uses the angular and linear velocities from the waypoint follower twist to calculate the corresponding steering angle.


## Results
By integrating all these modules, we can make the car follow waypoints on the road's middle lane while stopping in the presence of red lights.

![alt text][image1]

In order to see how the correct performance of the traffic light classifier, we set the car on manual mode for it to see red, yellow and green lights:

![alt text][image2]

Video of simulation on this [link](https://www.youtube.com/watch?v=gKV5OUGz5VY).

## Udacity Instructions

This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

Please use **one** of the two installation options, either native **or** docker installation.

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Port Forwarding
To set up port forwarding, please refer to the [instructions from term 2](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/16cf4a78-4fc7-49e1-8621-3450ca938b77)

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images

### Other library/driver information
Outside of `requirements.txt`, here is information on other driver/library versions used in the simulator and Carla:

Specific to these libraries, the simulator grader and Carla use the following:

|        | Simulator | Carla  |
| :-----------: |:-------------:| :-----:|
| Nvidia driver | 384.130 | 384.130 |
| CUDA | 8.0.61 | 8.0.61 |
| cuDNN | 6.0.21 | 6.0.21 |
| TensorRT | N/A | N/A |
| OpenCV | 3.2.0-dev | 2.4.8 |
| OpenMP | N/A | N/A |

We are working on a fix to line up the OpenCV versions between the two.
