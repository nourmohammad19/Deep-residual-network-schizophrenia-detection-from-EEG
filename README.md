# Deep-residual-network-schizophrenia-detection-from-EEG
A deep learning implementation of residual networks (ResNets) for automated schizophrenia detection using EEG signals. Based on the paper: [Exploring deep residual network based features for automatic schizophrenia detection from EEG].

link to the main paper : https://link.springer.com/article/10.1007/s13246-023-01225-8


## isntalation
1. clone the repository

   ```bash
   git clone [https://github.com/nourmohammad19/Deep-residual-network-schizophrenia-detection-from-EEG/edit/main/README.md](https://github.com/nourmohammad19/Deep-residual-network-schizophrenia-detection-from-EEG/edit/main/README.md)
   
   cd dl-rnn-vs-transformer-sequence-comparison

3. Install the required dependencies:

   ```bash
    pip install -r requirements.txt

4. Download the data

*  The data is available at https://www.kaggle.com/datasets/broach/button-tone-sz?resource=download
*  After downloading the data, please place it in the /data/csv folder
*  Then you should run the code in Data_transform.ipynb
*  Now your data is ready to be trained on

4. train the model
   
*  You can now easily train the model by running the train.py code
  
   ```bash
   cd src
   python train.py
     
