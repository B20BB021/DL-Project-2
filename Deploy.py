import streamlit as st
import pandas as pd
import numpy as np
from medpy.io import load
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torchvision
import pickle
import tempfile
import io
import torch.nn as nn
form resources.Unet import UNet




app_mode = st.sidebar.selectbox('Select Page',['Home','Give_Predictions'])
if app_mode=='Home': 
  st.title('Cancer something') 
  st.markdown('Dataset taken from the PI-CAI')

elif app_mode == 'Give_Predictions':
  st.subheader('Give image as a medpy supported file (e.g. .mha) :')
  FILE = st.file_uploader("FILE")
  
  with tempfile.NamedTemporaryFile(suffix = ".mha",delete=False) as tmp_file:
    if st.button("Predict"):
      fp = Path(tmp_file.name)
      fp.write_bytes(FILE.getvalue())
      image, header = load(fp.absolute())
      image = torch.from_numpy(image.transpose(2,0,1).astype(np.int16))		

      resizer = torchvision.transforms.Resize((104,104), antialias=False)
      resizer1 = torchvision.transforms.Resize((50,104), antialias=False)
      
      image = resizer(image)
      image = resizer1(image.permute(1,0,2)).permute(1,0,2).reshape(1,50,104,104)
      image = image.permute(1,0,2,3)
      
      model = UNet(1,1)

      picklefile = open("model.pkl", "rb")
      model = pickle.load(picklefile)
      img, prediction = model(image.float())
      
      
      
      for number in range(len(img)):
        
        fig, x = plt.subplots()
        x.imshow(img[number-1][0].detach().numpy(),cmap = "gray")
        st.caption("Label : ")
        st.caption(torch.argmax(prediction[number-1]).item())
        st.pyplot(fig)
