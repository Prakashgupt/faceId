
import os
import time
import random
import cv2
import pyrebase

from google.cloud import storage
from firebase_admin import credentials, initialize_app, storage
# from pyrebase import pyrebase

firebaseConfig = {
  "apiKey": "AIzaSyB4dBtz5u30QOWzGUMIesroxeaYYw0UAqU",
  "authDomain": "facedectectionst.firebaseapp.com",
  "projectId": "facedectectionst",
 "storageBucket": "facedectectionst.appspot.com",
  "messagingSenderId": "890307691112",
  "appId": "1:890307691112:web:d45ccdf4b7c5d7831f31dd",
  "measurementId": "G-S73JRLSSW1",
  "databaseURL": ""
}


firebase=pyrebase.initialize_app(firebaseConfig);
storage1=firebase.storage()
# st=storage1.list_files(self ='./')
# for file in st:
#     print(file.name)
#print(storage1.child().listAll())
#
# print(storage1.list_files())
# bucket=storage1.storage_bucket



# print(st)


# name_list = storage.list_files()
# for file in name_list:
#       print(file.name)


# Init firebase with your credentials
cred = credentials.Certificate("key.json")
st=initialize_app(cred, {'storageBucket': 'facedectectionst.appspot.com'})




# Put your local file path
def upload_blob(fileName):
    # fileName="./Train/Pravesh122/Pravesh122.jpg"


    store_path="Train/"+"orginal-"+str((int)(time.time())*random.randrange(0,100,2))+".jpg"
    bucket = storage.bucket()
    blob = bucket.blob(store_path)
    blob.upload_from_filename(fileName, content_type='image/jpg')
    print("file upload")
    # print(blob.public_url)
    url= storage1.child(store_path).get_url(None)
    return url


# print(bucket.path)
# it=bucket.list_blobs('Train')
# print(it.pages.send())
 # bucket.
# print(storage._utils)


# upload_blob("./Train/Pravesh122/Pravesh122.jpg")


def resizePhoto(path):
    img=cv2.imread(path)
    scale_percent = 50
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width , height)
    resized_img = cv2.resize(img, dim, cv2.INTER_AREA)
    os.remove(path)
    cv2.imwrite(path,resized_img)

    return

# resizePhoto("Train/_20220421_141356/1.jpg")

