#%%
import os,sys
parent_folder= "/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/"
sys.path.append(parent_folder)
from PIL import Image,ImageOps
import numpy as np
import matplotlib.pyplot as plt
import imageio
import torch
import cv2
#import imutils
import time
import dlib
import h5py
from itertools import repeat
# %%
video_path ="/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/data/PartA/vis/071313_m_41-PA4-072.mp4"
output_folder="/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/data/PartA/vis/"

#%%
def get_frames_list(video_path):
    frames_list=[]
    cap = cv2.VideoCapture(video_path) #video shape (1038, 1388)
    while True:
          # Read a frame from the video
          ret, frame = cap.read()
          if not ret:
              break
          image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
          frames_list.append(image)
    cap.release()
    return frames_list

def calcultate_events(frame_list,
                      threshold=0,
                      fps = 30,
                      size = (1038, 1388),
                      output_path='/content/drive/My Drive/Note Unifi/output.mp4',
                      stream_path='/content/drive/My Drive/Note Unifi/events_frames.npy'):
    event_frames=[]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
    for f in range(1,len(frame_list)):
      current_frame=np.array(frame_list[f])
      prev_frame=np.array(frame_list[f-1])
      #diff = cv2.subtract(current_frame, prev_frame)
      #diff=current_frame - prev_frame
      e_frame=np.zeros(current_frame.shape)
      e_normalized=np.zeros(current_frame.shape)
      for i in range(0,current_frame.shape[0]):
        for j in range(0,current_frame.shape[1]):
          if current_frame[i][j]>prev_frame[i][j]+threshold:
            e_frame[i][j]=1
          elif current_frame[i][j]<prev_frame[i][j]+threshold:
            e_frame[i][j]=-1
          e_normalized[i][j]=((e_frame[i][j]+1)/2 )*255
      event_frames.append(e_frame)
      out.write(np.uint8(e_normalized))
    out.release()
    np.save(stream_path,event_frames)
    return event_frames

def convert_and_trim_bb(image, rect):
    # extract the starting and ending (x, y)-coordinates of the
    # bounding box
    startX = rect.left()
    startY = rect.top()
    endX = rect.right()
    endY = rect.bottom()
    # ensure the bounding box coordinates fall within the spatial
    # dimensions of the image
    startX = max(0, startX)
    startY = max(0, startY)
    endX = min(endX, image.shape[1])
    endY = min(endY, image.shape[0])
    # compute the width and height of the bounding box
    w = endX - startX
    h = endY - startY
    # return our bounding box coordinates
    return (startX, startY, w, h)

def padding(img, desired_size):

    delta_width = desired_size[0] - img.shape[0]
    delta_height = desired_size[1] - img.shape[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(img.shape)
    p=cv2.copyMakeBorder(img, pad_height, pad_height, pad_width , pad_width ,cv2.BORDER_CONSTANT,value=[0,0,0])
    res=cv2.resize(p,desired_size)
    print(p.shape)
    return p  #ImageOps.expand(img, padding)

#Extract face from each frame
def extract_faces(video_path):
    detector = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(video_path) #video shape (1038, 1388)
    faces_list=[]
    max_width=0
    max_height=0
    while True:
          # Read a frame from the video
          ret, frame = cap.read()
          if not ret:
              break
          image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          rects = detector(image)
          boxes = [convert_and_trim_bb(image, r) for r in rects]
          # loop over the bounding boxes
          for (x, y, w, h) in boxes:
            # draw the bounding box on our image
            #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face=cv2.cvtColor(image[y:y+h,x:x+w], cv2.COLOR_BGR2GRAY)
            max_width=max(face.shape[0],max_width)
            max_height=max(face.shape[1],max_height)
            faces_list.append(face)
            #plt.imshow(face)
    cap.release()
    return faces_list,max_width,max_height # return gray frames

def get_paded_faces_list(faces_list,size):
    return list(map(padding,faces_list,repeat(size)))

def save_video_from_imgs(imgs,
                         size,
                         output_path,
                         fps=30,
                         colored=False):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    #tempVideo = TempFile(ext=".mp4")
    #writer = cv2.VideoWriter(tempVideo.path, fourcc, 30, (W, H), True)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (size[1], size[0]),colored)
    for f in range(1,len(imgs)):
        out.write(imgs[f])
    out.release()

#%%
faces_list,max_width,max_height=extract_faces(video_path)
#%%
save_video_from_imgs(faces_list,
                    size=[max_width,max_height],
                     output_path=output_folder+"faces_video_Gray.mp4",
                     colored=False)
#%%
paded_faces_list=get_paded_faces_list(faces_list,[max_width,max_height])


#%%
save_video_from_imgs(paded_faces_list,
                    size=[max_width,max_height],
                    output_path=output_folder+"padded_faces_video.mp4"
                    ,colored=False)
#%%

fram_faces_lists=get_frames_list("/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/data/PartA/vis/faces_video_Gray.mp4")
#%%
for i in fram_faces_lists:
    print(i.shape)
#%%
calcultate_events(fram_faces_lists,
                  threshold=0,
                  size=(max_width,max_height),
                  output_path=output_folder+'video_event_faces_th0.mp4',
                  stream_path=output_folder+'events_frames_faces_th0.npy')
#%%

plt.imshow(faces_list[0])
# %%
print([max_width,max_height])
# %%
