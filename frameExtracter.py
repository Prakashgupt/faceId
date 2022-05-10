

import dlib
import numpy as np

import cv2
import imutils
from imutils import face_utils
import os
import shutil
import math
from PIL import Image


model_path = 'models//shape_predictor_68_face_landmarks.dat'
faceLandMarkDetector = dlib.shape_predictor(model_path)
detected_landmarks = dlib.shape_predictor(model_path)
frontalFaceDetector = dlib.get_frontal_face_detector()
kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))


def extract_eye_region(shape,image):
    (x,y,w,h) = cv2.boundingRect(np.array([shape[17:47]]))
    #roi = image[y  : y + (2*h)//3 ,  x : x + w]
    roi = image[y - h//2 : y + (2*h)//3 ,  x : x + w]
    #roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    return roi


def mark_landmarks(Face, image):
    if len(Face) == 0:
      print('No face detected !')
      return None
    elif len(Face) >= 2:
      print('multiple face detected !')
      return None
    else:
      for (i, rect) in enumerate(Face):
        #print(i, rect)
        shape = detected_landmarks(image, rect)
        shape = face_utils.shape_to_np(shape)
        return shape
def euclidean_distance(a, b):
    x1 = a[0]; y1 = a[1]
    x2 = b[0]; y2 = b[1]
    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

def alignFace(Face, image):
    for (i, rect) in enumerate(Face):
        shape = detected_landmarks(image, rect)
        shape = face_utils.shape_to_np(shape)

        LeftEye = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
        RightEye = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
        #store = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
       # print(LeftEye, RightEye)
        
        left_eye_coord = (shape[LeftEye[0]], shape[LeftEye[1]])
        right_eye_coord = (shape[RightEye[0]], shape[RightEye[1]])
      #  print(left_eye_coord)
        
        
        leftEyePts = shape[LeftEye[0]:LeftEye[1]]
        rightEyePts = shape[RightEye[0]:RightEye[1]]
        
        leftEyeCenter = leftEyePts.mean(axis = 0).astype('int')
        rightEyeCenter = rightEyePts.mean(axis = 0).astype('int')
       # print('leftEyeCenter', leftEyeCenter, 'rightEyeCenter', rightEyeCenter)
        
        if leftEyeCenter[1] < rightEyeCenter[1]:
            point_3rd = (rightEyeCenter[0], leftEyeCenter[1])
            #rotate clockwise
            direction = -1 
        else:
            point_3rd = (leftEyeCenter[0], rightEyeCenter[1])
            #rotate anti-clockwise
            direction = 1
        
        a = euclidean_distance(leftEyeCenter, point_3rd)
        b = euclidean_distance(rightEyeCenter, leftEyeCenter)
        c = euclidean_distance(rightEyeCenter, point_3rd)
        cos_a = c/b
        angle = np.arccos(cos_a)
        angle = (angle * 180) / math.pi
        
       # print(direction)
        
        if direction == -1:
            angle = 90 - angle
        #elif direction == 1:
        #    angle =  angle - 90
        
        alignedFace = Image.fromarray(image)
        alignedFace = np.array(alignedFace.rotate(direction * angle))
        return alignedFace

def image_to_eyes(image_path,title,img_name,store_folder):
      # image_pp = os.path.join(image_folder_path, 'image')
  # store_folder = './Train'
  path=os.path.join(store_folder,title)
  if not os.path.exists(path):
        os.mkdir(path)


  # listt = os.listdir(image_pp)
  # if(len(listt) == 0):
  #   print('empty test folder  \n\nEXITING EXECUTION')
  #   return
  # image_path = os.path.join(image_pp, listt[0])
  actual_image =  cv2.imread(image_path)
  resized_actual_image = imutils.resize(actual_image, width = 480)
  
  Face = frontalFaceDetector(resized_actual_image , 1)
  if(len(Face) == 0) :
    shutil.rmtree(path)
    print("no face detected!")
    return 
  
  aligned_face = alignFace(Face , resized_actual_image)

  Face2 = frontalFaceDetector(aligned_face, 1)
  shape2 = mark_landmarks(Face2, aligned_face)
  if shape2 is not None:
    eye_region = extract_eye_region(shape2 , aligned_face)
    eye_region=cv2.cvtColor(eye_region,cv2.COLOR_RGB2GRAY)
    if eye_region.shape[0] != 0 and eye_region.shape[1] != 0:
        top_hat=cv2.morphologyEx(eye_region,cv2.MORPH_TOPHAT,kernel)
        black_hat=cv2.morphologyEx(eye_region,cv2.MORPH_BLACKHAT,kernel)
        eye_region=eye_region+top_hat-black_hat
        print(eye_region.shape)
        file_name = os.path.join(path,img_name )
        cv2.imwrite(file_name, eye_region)
        return eye_region
    else:
        shutil.rmtree(path)
        print("FACE NOT DETECTED!!")
        return
  else:
    shutil.rmtree(path)
    print( "FACE NOT DETECTED!!")
    return

# image_to_eyes("Images/Prakash1/Prakash1.png","Prakash1")
def chek1(image_path,title):
    store_folder = './Checkeye'
    # req=urllib.request.urlopen(url)
    # arr = np.asarray(bytearray(req.read()))
    # img=cv2.imdecode(arr, cv2.IMREAD_COLOR)
    # print(img.shape)

    eye_img=image_to_eyes(image_path,title,'extracted_eye.jpg',store_folder)
    return eye_img



def chek(image_path,title,img_name):
    store_folder = './Addeye'
    # req=urllib.request.urlopen(url)
    # arr = np.asarray(bytearray(req.read()))
    # img=cv2.imdecode(arr, cv2.IMREAD_COLOR)
    # print(img.shape)
    eye_img=image_to_eyes(image_path,title,img_name,store_folder)
    return eye_img

#chek("https://firebasestorage.googleapis.com/v0/b/facedectectionst.appspot.com/o/Images%2FPARVESH1a%2FPARVESH1a.jpg?alt=media&token=af858a81-a2b4-43b4-8a49-fd62fb9d701d",'kajol12')



