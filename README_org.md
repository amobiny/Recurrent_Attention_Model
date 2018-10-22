# Recurrent Models of Visual Attention 

![glimpse demo](https://github.com/PrincipalComponent/ram/blob/master/figures/mnih2014/cluttered60x60_6glimpses/cluttered_demo.gif)
![glimpse demo](https://github.com/PrincipalComponent/ram/blob/master/figures/cluttered_var/cluttered_var100x10_4-20.gif)
![glimpse demo](https://github.com/PrincipalComponent/ram/blob/master/figures/cluttered_var/200x200_conv_4-50.gif)
![glimpse demo](https://github.com/PrincipalComponent/ram/blob/master/figures/cluttered_var/200x200_conv_20-100.gif)

Contains my implementation of the RAM model originally developed by [1] and [2]. 

While the first version from [1] was extended to handle multiple objects in [2], the implementation here reproduces [1].

# Objectives

1. replicate all RAM results on the translated and cluttered MNIST task (60x60 and 100x100). [x]
2. compare to baselines, (a) fully connected and (b) CNNs, matched in number of params. [x]
3. replicate all DRAM results from [2] [x]

# Results

* replicated all results from [1]: 28x28 MNIST and translated/cluttered MNIST (60x60 pixels)

### Translated 60x60

| Model                                 | Error       |
| -------------                         |-------------|
| RAM, 6 glimpses, 12x12, 3 scales      |  1.76%      |
| RAM, 8 glimpses, 12x12, 3 scales      |  2.12%      |
| DRAM, 6 glimpses, 12x12, 2 scales     | --          |
| DRAM, 8 glimpses, 12x12, 2 scales     | --          |

### Cluttered 60x60

| Model                                 | Error       |
| -------------                         |-------------|
| RAM, 6 glimpses, 12x12, 3 scales      | 3.75%       |
| RAM, 8 glimpses, 12x12, 3 scales      | 4.82%       |
| DRAM, 6 glimpses, 12x12, 2 scales     | 2.75%       |
| DRAM, 8 glimpses, 12x12, 2 scales     | --          |

### Cluttered 100x100

| Model                                 | Error       |
| -------------                         |-------------|
| RAM, 6 glimpses, 12x12, 3 scales      | 4.62%       |
| RAM, 8 glimpses, 12x12, 3 scales      | 2.68%       |
| DRAM, 6 glimpses, 12x12, 2 scales     | 2.00%       |
| DRAM, 8 glimpses, 12x12, 2 scales     | 1.71%       |

### Cluttered 250x250

| Model                                 | Error       |
| -------------                         |-------------|
| DRAM, 8 glimpses, 20x20, 2 scales     | 12.2%       |


## Architectures & training details

* ReLU activation function in all modules
* Adam optimizer, exponential learning rate decay: 0.001 - 0.0001, 1750 steps (gamma: 0.97)
* Nr. of mini-batches: 60000 (RAM), 100000 (DRAM)
* Monte-Carlo samples: N=10
* Standard deviation of location samples: 0.11


#### RAM

| Module                                | Layers      |
| -------------                         |-------------|
| Glimpse network -- location           |  FC: 2 -> 128 -> 256   |
| Glimpse network -- glimpse            |  FC: 144 -> 128 -> 256  |
| Location network                      |  FC: 256 -> 2    |
| Classification network                |  FC: 256 -> 10   |
| Context network                       |  None             |
| Core recurrent network                |  LSTM: 256 -> {1 layer, 256 units} -> 256|

#### DRAM

| Module                                | Layers      |
| -------------                         |-------------|
| Glimpse network -- location           |  FC: 2 -> 128 -> 256   |
| Glimpse network -- glimpse            |  FC: 144 -> 128 -> 256  |
| Location network                      |  FC: 256 -> 2    |
| Classification network                |  FC: 256 -> 10   |
| Context network                       |  CNN: 24x24 -> {5x5, 32} -> {3x3, 32} -> FC: 256 |
| Core recurrent network                |  LSTM: 256 -> {2 layers, 256 units} -> 256|


# L-DRAM: including location loss

### Cluttered 100x100, 30 000 updates

| Model                                 | % labels    | Steps to 90%| Error       |
| --------------------------------------|-------------|-------------|-------------|
| L-DRAM, 6 glimpses, 12x12, 2 scales   | 1%          |             |  2.59%      |
| L-DRAM, 6 glimpses, 12x12, 2 scales   | 5%          |             |  2.06%      |
| L-DRAM, 6 glimpses, 12x12, 2 scales   | 10%         |             |  2.68%      |
| L-DRAM, 6 glimpses, 12x12, 2 scales   | 25%         |             |  2.12%      |
| L-DRAM, 6 glimpses, 12x12, 2 scales   | 50%         |             |  2.10%      |

### Cluttered 200x200, variable # distractions (4-50), 30 000 updates

| Model                                 | % labels    | Steps to 90%| Error       |
| --------------------------------------|-------------|-------------|-------------|
| L-DRAM, 4 glimpses, 12x12, 2 scales   | 100%        |             |  5.26%      |



# TODO

1. MNIST data sets with location ground truth   [x]
2. location loss                                [x]
3. summarize speed-up on cluttered MNIST        [x]
4. plot learning curves on cluttered MNIST      [x]
5. exponential decay of location loss importance[x]

---

[1] Mnih V., Heess N., Graves A., Kavukcuoglu K. 2014. Recurrent Models of Visual Attention. <https://arxiv.org/abs/1406.6247>
[2] Ba J., Mnih V., Kavukcuoglu K. 2014. Multiple Object Recognition with Visual Attention. <https://arxiv.org/abs/1412.7755>
