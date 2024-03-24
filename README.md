# handwritingdigit

## 1 Introduction
A handwritten digit recognition neural network visual demonstration program

![UI](Screenshot.png)

## 2 Dependence
* PyTorch https://pytorch.org/
* NumPy https://numpy.org/
* Jupyter https://jupyter.org/
* matplotlib https://matplotlib.org/

## 3 Before you start 
Install **pytorch**, **numpy** first. Please select the installation method according to your own situation. For pytorch, please refer to https://pytorch.org/get-started/locally/ 

If you need to run or debug the notebook, you need to install the **jupyter**, **matplotlib** program.

or you can simply:
```
pip install numpy
pip install matplotlib
pip install jupyterlab
pip install torch torchvision torchaudio
```

## 4 File list

```
.
├── hwdigit.py                  # visual demonstration program
├── load_model_predict.ipynb    # a notebook for load model and predict test data set
├── load_model_predict.pdf      # pdf version
├── model_predict.pth           # the model we train and saved
├── README.md
├── Screenshot.png
├── train_with_data.ipynb       # a notebook for train model
└── train_with_data.pdf         # pdf version
```

## 5 Start the demonstration
```
python hwdigit.py 
```