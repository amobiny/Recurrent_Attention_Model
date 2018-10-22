# Location-guided deep recurrent attention models (LG-DRAM)

Contains the code for my thesis project on location-guided recurrent attention models (LG-DRAM). 

In this project I developed a training method for stochastic recurrent attention models [1][2] that enhances both recognition performance and learning speed.

## Abstract

Similar to how humans direct their gaze, stochastic recurrent attention models process images in a sequence of glimpses focusing on the parts of the input most relevant for the task at hand, e.g. object detection. Trained with reinforcement learning they learn where to look in order to maximise their performance in a goal-driven way. However, parameter optimisation can be slow especially for large cluttered images.

We therefore propose to enrich the training procedure with an auxiliary supervised learning task, namely object localisation. Like a teacher occasionally pointing a student towards the relevant regions in an image this additional task term strengenths the reward signal and biases glimpses towards the target objects. Crucially, this method only requires very few location-annotations and is therefore useful in practice to make attention models more data efficient.


## Demo

Samples of the model evaluated on several digit classification tasks (boxes indicate the glimpses; green: correct classification, red: incorrect).

### Cluttered MNIST (CMNIST)
![glimpse demo](https://github.com/PrincipalComponent/ram/blob/master/figures/mnih2014/cluttered60x60_6glimpses/cluttered_demo.gif)

### Modified cluttered MNIST (M-CMNIST)
![glimpse demo](https://github.com/PrincipalComponent/ram/blob/master/figures/cluttered_var/cluttered_var100x10_4-20.gif)
![glimpse demo](https://github.com/PrincipalComponent/ram/blob/master/figures/cluttered_var/200x200_conv_4-50.gif)
![glimpse demo](https://github.com/PrincipalComponent/ram/blob/master/figures/cluttered_var/200x200_conv_20-100.gif)

---

[1] Mnih V., Heess N., Graves A., Kavukcuoglu K. 2014. Recurrent Models of Visual Attention. <https://arxiv.org/abs/1406.6247>
[2] Ba J., Mnih V., Kavukcuoglu K. 2014. Multiple Object Recognition with Visual Attention. <https://arxiv.org/abs/1412.7755>