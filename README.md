# KAN_GCN_Traffic
With the spirit of reproducible research, this repository contains codes required to produce the results in the manuscript:

> J. Zhang, Y. Zhang, Y. Zheng, Y. Wang, J. You, Y. Xu, W. Jiang, S. Dev, TrafficKAN-GCN: Graph Convolutional-based Kolmogorov-Arnold Network for Traffic Flow Optimization, *Decision Analysis Journal*, Under review.

## Citing
If you find our model useful in your research, please consider citing our [paper](There should have been a mysterious link here).
```
Waiting for review ~
```
This code is only for academic and research purposes.

### Executive summary



### Pre-requisites
```
Python 3.9.7 or higher
Or you can run the above code on AutoDL, the environment I use is:
GPU RTX 2080 Ti (11GB) * 1
CPU 12 vCPU Intel(R) Xeon(R) Platinum 8255C CPU @ 2.50GHz
```

### Requirements

```python
# python==3.9.7
matplotlib==3.6.2
numpy==1.24.4
scikit_learn==1.1.3
setuptools==65.5.0
sympy==1.11.1.
torch==2.2.2
tqdm==4.66.2
pandas==2.0.1
seaborn
pyyaml
```
### Data
The data source in this work is “https://baltometro.org/about-us/datamaps/regional-gis-data-center” and "https://opendata.baltimorecountymd.gov".

### Author's note
The version uploaded here is applied to the Cora dataset because the traffic related data files are too large to upload to GitHub. You can download the dataset by yourself from the MCM/ICM official website, we used the data related to the Problem D.


### Advice on hyperparameter tuning
Many intuitions about MLPs and other networks may not directly transfer to KANs. So how can I tune the hyperparameters effectively? Here is my general advice based on my experience playing with the problems reported in the paper. Since these problems are relatively small-scale and science-oriented, it is likely that my advice is not suitable to your case. But I want to at least share my experience so that users can have better clues on where to start and what to expect from tuning hyperparameters.

Start with a simple setup (small KAN shape, small grid size, small data, no regularization lamb=0). This is quite different from the MLP literature, where widths of order O(10^2) or larger are common. For instance, if you have a task with 5 inputs and 1 output, you could start with something like KAN(width=[5,1,1], grid=3, k=3). If that doesn’t work, first gradually increase the width. If that still fails, then consider increasing the depth. You don’t need to be this conservative if you already have a better sense of your task’s complexity.

*Once you achieve acceptable performance, you can further refine your KAN (either for better accuracy or improved interpretability).

*If you care about accuracy, try the grid extension technique. An example is here. But be careful about overfitting, see below.

*Once you obtain reasonably good results, try increasing the data size and running a final training step, which should further improve performance.

*Disclaimer: Starting with the simplest setup first is a mindset borrowed from physics, which is somewhat personal/biased, but I’ve found it effective and easier to control. Another reason I prefer small initial datasets is to get quicker feedback during debugging (especially since my early implementations were slow). This approach assumes that small datasets behave qualitatively similarly to larger datasets, which isn’t always true but tends to hold in the small-scale problems I’ve worked with. To check if your data is sufficient, see the next paragraph.

Another useful thing to keep in mind is to constantly check whether your model is underfitting or overfitting. If there is a large gap between training and testing losses, you may want to increase data size or reduce model complexity (grid size is more critical than width, so try reducing grid first, then width). This is also why I prefer to start with simple models — to ensure the model starts in the underfitting regime and gradually expands into the “optimal zone.”
