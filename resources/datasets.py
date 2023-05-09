import os
import torchvision
import pandas as pd
from torch.utils.data import Dataset


class dataset(Dataset):
  def __init__(self, img_dir, label_dir, marksheet, transform=None, target_transform=None, resizeLen = (25,100,100), fileI = None, fileL = None):
    
    self.img_names = []
    self.label_dir = label_dir
    for entry in os.scandir(label_dir):
      self.img_names.append(entry.name[:-7])
    self.marksheet = pd.read_csv(marksheet)
    self.marksheet = self.marksheet.set_index("study_id")["case_ISUP"]
    self.resizer = torchvision.transforms.Resize((resizeLen[1],resizeLen[2]), antialias=False)
    self.resizer2 = torchvision.transforms.Resize((resizeLen[0],resizeLen[2]), antialias=False)
    self.img_dir = img_dir
    self.transform = transform
    self.target_transform = target_transform

    self.storeI = []
    self.storeL = []
    if not (fileI and fileL):
      for idx in range(len(self.img_names)):
        
        img_path = self.img_dir + '/' + self.img_names[idx][:5] + '/'+ self.img_names[idx] +'_t2w.mha'
        label_path = self.label_dir+ '/' + self.img_names[idx] + '.nii.gz'
        image, header = load(img_path)
        label, label_header = load(label_path)
        image = torch.from_numpy(image.transpose(2,0,1).astype(np.int16))
        label = torch.from_numpy(label.transpose(2,0,1).astype(np.int16))
        image = self.resizer(image)
        image = self.resizer2(image.permute(1,0,2)).permute(1,0,2).reshape(1,resizeLen[0],resizeLen[1],resizeLen[2]).numpy()
        label = self.resizer(label)
        label = self.resizer2(label.permute(1,0,2)).permute(1,0,2).reshape(1,resizeLen[0],resizeLen[1],resizeLen[2]).numpy()
        self.storeI.append(image)
        self.storeL.append(label)
      self.storeI = np.concatenate(self.storeI)
      self.storeL = np.concatenate(self.storeL)
    else:
      self.storeI = np.load(fileI)
      self.storeL = np.load(fileL)
  def __len__(self):
    return len(self.img_names)
  
  def __getitem__(self, idx, permute = (0,1,2)):
    ISUP_case = self.marksheet[int(self.img_names[idx][-7:])]
    image = torch.from_numpy(self.storeI[idx])
    label = torch.from_numpy(self.storeL[idx])
    if self.transform:
      image = self.transform(image)
    if self.target_transform:
      label = self.target_transform(label)

    return image.permute(permute), label.permute(permute), ISUP_case
    
class dataset_OLD():
  def __init__(self, img_dir, label_dir, marksheet, transform=None, target_transform=None, resizeLen = 500):
    
    self.img_names = []
    self.label_dir = label_dir
    self.resizeLen = resizeLen
    for entry in os.scandir(label_dir):
      self.img_names.append(entry.name[:-7])

    self.marksheet = pd.read_csv(marksheet)
    self.marksheet = self.marksheet.set_index("study_id")["case_ISUP"]
    
    self.img_dir = img_dir
    self.transform = transform
    self.target_transform = target_transform
  def length(self):
    return len(self.img_names)
  def getitem(self, idx):
    label_path = self.label_dir+ '/' + self.img_names[idx] + '.nii.gz'
    img_path = self.img_dir + '/' + self.img_names[idx][:5] + '/'+ self.img_names[idx] +'_t2w.mha'
    ISUP_case = self.marksheet[int(self.img_names[idx][-7:])]
    image, header = load(img_path)
    label, label_header = load(label_path)
    image = torch.from_numpy(image.transpose(2,0,1).astype(np.int16))
    label = torch.from_numpy(label.transpose(2,0,1).astype(np.int16))
    resizer = torchvision.transforms.Resize((self.resizeLen,self.resizeLen), antialias=False)
    image = resizer(image)
    label = resizer(label)
    if self.transform:
      image = self.transform(image)
    if self.target_transform:
      label = self.target_transform(label)
    return image, label, ISUP_case
