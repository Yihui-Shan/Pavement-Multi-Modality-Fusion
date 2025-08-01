## 🌐 Usage

### ⚙ Network Architecture

Our net is implemented in ``net.py``.

### 🏊 Training
**1. Virtual Environment**
```
# create virtual environment
conda create -n cddfuse python=3.8.10
conda activate cddfuse
# select pytorch version yourself
# install cddfuse requirements
pip install -r requirements.txt

**2. Data Preparation**

 Place it in the folder ``'./train_img/'``.
**3. Pre-Processing**
Run 
```
python dataprocessing.py
``` 
and the processed training dataset is in ``'./data/PMMF_train_imgsize_128_stride_200.h5'``.
**4. CDDFuse Training**
Run 
```
python train.py
``` 
and the trained model is available in ``'./models/'``.
### 🏄 Testing

**1. Pretrained models**

Pretrained models are available in ``'./models/PMMF.pth'``, which are responsible for the Depth-Intensity Fusion  tasks.

**2. Test datasets**

The test datasets used in the paper have been stored in ``'./test_img/CMPV'``.

