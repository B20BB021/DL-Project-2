import requests
from zipfile import ZipFile
import subprocess


class picai_handler:
  
  def __init__(self, path = '..', howMany = 5):
    self.howMany = howMany
    if(self.doPaths(path) == -1):
      return None
    if(self.download_zips() == -1):
      return None
    self.extract_zips()
    for i in range(howMany):
      rsyncCMD = "rsync -a /content/drive/MyDrive/picai-challenge-data/unzipped/picai_public_images_fold"+str(i)+"/ /content/drive/MyDrive/picai-challenge-data/merged_unzipped/"
      subprocess.run(rsyncCMD, shell = True)
  
  def doPaths(self, path):
    self.zipPaths = {}
    self.dataFolder = ''
    self.zipFolder = ''
    if (path[-1] == '/'):
      path = path[:-1]
    if (path[0] != '/'):
      path = '/'+path
    if not (os.path.isdir(path)):
      print("Path is not a directory.")
      return -1
    self.dataFolder = path + '/picai-challenge-data'

    if not (os.path.isdir(self.dataFolder)):
      os.mkdir(self.dataFolder)
    self.zipFolder = self.dataFolder+'/zips'
    if not (os.path.isdir(self.zipFolder)):
      os.mkdir(self.zipFolder)
    self.unzippedFolder = self.dataFolder+"/unzipped"
    if not (os.path.isdir(self.unzippedFolder)):
      os.mkdir(self.unzippedFolder)
    self.unzippedmerged = self.dataFolder+"merged_unzipped"
    if not (os.path.isdir(self.unzippedmerged)):
      os.mkdir(self.unzippedmerged)
    for x in range(self.howMany):
      self.zipPaths[x] = ('picai_public_images_fold'+str(x)+'.zip')
    self.zipPaths["license"] = "LICENSE"
    self.zipPaths["readme"] = "README.md"
  
  def download_zips(self):
    howManyZips = self.howMany
    if(howManyZips > 5):
      print("Too many ZIP files specified. Actual PICAI only has max of 5.")
      return -1
    for key, fileName in self.zipPaths.items():
      if (isinstance(key,int)):
        if key < howManyZips:
          pass
        else:
          continue
      totName = self.zipFolder+'/'+fileName
      if not(os.path.isfile(totName)):
        print("Downloading "+fileName)
        File = requests.get('https://zenodo.org/record/6624726/files/'+fileName+'?download=1', stream = True) 
        with open(totName, "wb") as fule: 
          for block in File.iter_content(chunk_size = 1024):
            if block: 
              fule.write(block)
        print("Downloaded "+fileName)
      else:
        print(fileName + " already exists in " + self.zipFolder)
       
  def extract_zips(self):

    for x in range(self.howMany):
      
      if not(os.path.isdir(self.unzippedFolder + '/' + self.zipPaths[x][:-4])):
        with ZipFile(self.zipFolder+'/'+ self.zipPaths[x], 'r') as zap:
          print("Extracting " + self.zipPaths[x])
          zap.extractall(self.unzippedFolder + '/' + self.zipPaths[x][:-4])
          print('Done!')
      else:
        print(self.unzippedFolder + '/' + self.zipPaths[x][:-4]+' already exists in', self.unzippedFolder)
    return 0
