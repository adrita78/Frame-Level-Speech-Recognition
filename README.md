# Frame-Level-Speech-Recognition

This Jupyter notebook contains the implementation of a multi-layer perceptron using Python programming language. The model was developed to solve a specific problem or task, which will be described below.

# Problem Statement
In this competition, we created a multilayer perceptron for frame-level speech recognition. Specifically,a frame-level phonetic transcription of raw Mel Frequency Cepstral Coefficients (MFCCs) was created. The MLP learnt a feature representation and nonlinear classification boundary. To discriminate between each phoneme class label, cross-entropy loss was used to minimize the dissimilarity between output logits and the target labels. The loss was used during gradient descent to update the parameters of the neural network, thereby minimizing the cost function.

# Datatset
 The training data comprises of:
• Speech recordings (raw mel spectrogram frames)
• Frame-level phoneme state labels
The test data comprises of:
• Speech recordings (raw mel spectrogram frames)
• Phoneme state labels are not given

One can download the dataset from the Kaggle website at the following link: https://www.kaggle.com/competitions/11-785-s23-hw1p2/data Once downloaded, extract the contents of the zip file and store the data in a directory named data. The dataset can directly downloaded into your python notebook.

# Dependencies

Python 3.6+
PyTorch 2.0
Numpy
Matplotlib
Wandb
DataLoader,TensorDataset

# Running the Code
Once you have installed the dependencies and downloaded the dataset, you can run the code by opening the Jupyter Notebook "Train-Clean-360.ipynb" in your Jupyter Notebook environment. 

The notebook contains the following sections:

1. Memory Effcient Data Loading
2. Defining the AudioDataset Class
4. Model Architecture
5. Hyperparameter Tuning
6. Training the Model
7. Evaluating the Model
8. Hyperparameter Tuning

# Experiments



# Conclusion




