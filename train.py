from torch.utils.data import DataLoader
form resources.datasets import dataset 
form resources.Unet import UNet
import torch 
import pickle

merged_unzipped = './picai-challenge-data/merged_unzipped'
lesion_delin = './picai-challenge-data/picai_labels/csPCa_lesion_delineations/AI/Bosma22a'
marksheet_p = './picai-challenge-data/picai_labels/clinical_information/marksheet.csv'
fileI = "./picai-challenge-data/50_104_104I.npy"
fileL = "./picai-challenge-data/50_104_104L.npy"
dataset = dataset(merged_unzipped, lesion_delin,marksheet_p, None, None, resizeLen = (50,104,104), fileI = fileI, fileL = fileL)
loader = DataLoader(dataset, batch_size=1, shuffle=True)



encoder = UNet(1,1).to(device)
encoder.train()
encoderO = torch.optim.Adam(encoder.parameters())

def replace(new_val, lst):
  ls = []
  for i in lst:
    if i>0:
      ls.append(int(new_val))
    else:
      ls.append(int(i))
  return ls




epochs = 20

for epoch in range(1,epochs):
  train_loss = 0
  z = 0
  acc = 0
  for x, batch in enumerate(loader):
    images, labeled, ISUP_case = batch
    
    # print( k)
    if(torch.sum(labeled) == 0):
      if(np.random.rand() <  0.7):
        continue

    images = images.to(device).float()
    labeled = labeled.to(device).float()
    
    # ISUP_case = nn.functional.one_hot(ISUP_case).to(device)
    images = images.permute(1,0,2,3)
    labeled = labeled.permute(1,0,2,3)
    k = [labeled[i].max() for i in range(len(labeled))]
    # print(k)
    k = replace(int(ISUP_case), k)
    k = torch.as_tensor(k).to(device)
    encoderO.zero_grad()
    
    # labelO, reconO, out = encoder(images)
    out,ISUPO = encoder(images)
    loss = bce_dice_loss(out,labeled)+nn.functional.cross_entropy(ISUPO, k)
    # + nn.functional.mse_loss(labelO, labeled) + nn.functional.cross_entropy(ISUPO, ISUP_case)
    loss.backward()
    acc += torch.sum(torch.argmax(ISUPO, 1) == k).item()
    # print(torch.sum(torch.argmax(ISUPO, 1) == k))
    train_loss+=loss.detach().item()
    z+=1
    encoderO.step()

  print("\nEpoch: ",epoch," loss : ",float(train_loss/z), "Accuracy: ", acc/(50*z)*100)
  
  

pickle_out = open("./model.pkl","wb")
pickle.dump(encoder, pickle_out)
pickle_out.close()
