# DL-Project-2

To easily download the data from picai, create an instance of picai-handler (from resources/picai.py) (download.py does this). It will slowly download, unzip and place the dataset where it needs to go.

The model was trained using the training loop in train.py. The saved model is the one used in Deploy.py. It needs to be in the same directory as Deploy.py.

The streamlit code for this is in Deploy.py.

The libraries required for train.py are - 
  medpy
  torch
  torchvision
  numpy
  pandas
  os
  pickle
  
The libraries required for Deploy.py are -
  
  
The libraries required for download.py are - 
  requests
  ZipFile
  os
  subprocess

You can install all of the above using pip install LibraryName

Running download.py from this directory should install all the data in the right place for train.py.
So just use python download.py to download the data.
