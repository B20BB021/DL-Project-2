# DL-Project-2

To easily download the data from picai, create an instance of picai-handler (from resources/picai.py) (download.py does this). It will slowly download, unzip and place the dataset where it needs to go.

The model was trained using the training loop in train.py. The saved model is the one used in Deploy.py. It needs to be in the same directory as Deploy.py.

The streamlit code for this is in Deploy.py.
We took the model from given github https://github.com/s0mnaths/Brain-Tumor-Segmentation and modified it to make it for multitasking learning. This model gives labelled image with tumor masked with white region and also give the ISUP case.

<h1> Dependencies </h1>
The libraries required for train.py are - 
<ol>
  <li>medpy</li>
  <li>torch</li>
  <li>torchvision</li>
  <li>numpy</li>
  <li>pandas</li>
  <li>os</li>
  <li>pickle</li>
</ol>
The libraries required for Deploy.py are -
  

The libraries required for download.py are - 
<ol>
  <li>requests</li>
  <li>ZipFile</li>
  <li>os</li>
  <li>subprocess</li>
</ol>
You can install all of the above using 

```sh
pip install LibraryName
```

Running download.py from this directory should install all the data in the right place for train.py.
So just use the following. 

```sh
python download.py 
```

To train the model - 

```sh
python train.py 
```

The entire code that we used in the form of a notebook is present in notebook.ipynb.

Technically you don't need any of this, only the model is required.

You can download the model directly from the following link - <a href = ''>LINK</a>
