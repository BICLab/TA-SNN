# Temporal-wise Attention Spiking Neural Networks for Event Streams Classification

### **Configuration requirements**
1. Python 3.7.4
2. PyTorch 1.7.1
3. tqdm 4.56.0
4. numpy 1.19.2



### **Instructions**
#### 1. DVS128 Gesture
1. Download [DVS128 Gesture](https://www.research.ibm.com/dvsgesture/) and put the downloaded dataset to /DVSGestures/data, then run /DVSGestures/data/DVS_Gesture.py.
```
TA_SNN
├── /DVSGestures/
│  ├── /data/
│  │  ├── DVS_Gesture.py
│  │  └── DvsGesture.tar.gz
```
2. Change the values of T and dt in /DVSGestures/CNN/Config.py or /DVSGestures/CNN_10clips/Config.py then run the tasks in /DVSGestures.

eg:
```
python SNN_CNN.py
```
3. View the results in /DVSGestures/CNN/Result/ or /DVSGestures/CNN_10clips/Result/.



#### 2. CIFAR10-DVS
1. Download [CIFAR10-DVS](https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671/2) and processing dataset using official matlab program, then put the result to /CIFAR10DVS/data.
```
TA_SNN
├── /CIFAR10DVS/
│  ├── /data/
│  │  ├── /airplane/
│  │  |  ├──0.mat
│  │  |  ├──1.mat
│  │  |  ├──...
│  │  ├──automobile
│  │  └──...
```
2. Change the values of T and dt in /CIFAR10DVS/CNN/Config.py or /CIFAR10DVS/CNN_10clips/Config.py then run the tasks in /CIFAR10DVS.

eg:
```
python SNN_CNN.py
```
3. View the results in /CIFAR10DVS/CNN/Result/ or /CIFAR10DVS/CNN_10clips/Result/.




#### 3. SHD Dataset
1. Download [SHD Dataset](https://compneuro.net/datasets/) and put the downloaded dataset to /SHD/data.
```
TA_SNN
├── /SHD/
│  ├── /data/
│  │  ├── shd_train.h5
│  │  └── shd_test.h5
```
2. Change the values of T and dt in /SHD/MLP/Config.py then run the tasks in /SHD.

eg:
```
python SNN_MLP_3.py
```
3. View the results in /SHD/MLP/Result/.
#### 4. Extra
1. /module/TA.py defines the Temporal Attention layer and /module/LIF.py,LIF_Module.py defines LIF module.
