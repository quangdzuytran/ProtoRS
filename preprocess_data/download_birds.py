import requests
import tarfile
import os
# import gdown

url = 'https://data.caltech.edu/tindfiles/serve/1239ea37-e132-42ee-8c09-c383bb54e7ff/'
# url = 'https://drive.google.com/uc?id=1hbzc_P1FuxMkcabkgn9ZKinBwW683j45' 
target_path = './data/CUB_200_2011/CUB_200_2011.tgz' #you might need to change the name to CUB_200_2011.tgz

print(os.path.abspath(target_path))

response = requests.get(url, stream=True, allow_redirects=True)
if response.status_code == 200:
    with open(target_path, 'wb') as f:
        f.write(response.raw.read())

#or download it manually and uncomment the line below
# gdown.download(url, target_path, quiet=False)
 
tar = tarfile.open(target_path, "r:gz")
tar.extractall(path='./data')
tar.close()
print("CUB downloaded")
