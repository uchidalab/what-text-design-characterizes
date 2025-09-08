import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import cv2
import pandas as pd
import numpy as np
import random
import torchvision
import os
from img_preprocessing import preprocessing
import glob
import gensim
import re
from PIL import Image, ImageDraw, ImageFont
from models import AutoEncoder,ResNet50
import pickle
random.seed(0)

class Mydatasets(torch.utils.data.Dataset):
    def __init__(self,word_vec_list,label_list,height_list,coord_list,all_word_list,transform = None):
        self.transform = transform
        self.word_vec_list = word_vec_list
        self.label_list = label_list
        self.height_list = height_list
        self.coord_list = coord_list
        self.all_word_list = all_word_list
        self.datanum = len(label_list)
    def __len__(self):
        return self.datanum
    
    def __getitem__(self, idx):
        vec = {}
        vec["semantic"]  = self.word_vec_list[idx][:,0,:].unsqueeze(1)
        vec["font_style"] = self.word_vec_list[idx][:,1,:].unsqueeze(1)
        vec["char_color"] = self.word_vec_list[idx][:,2,:].unsqueeze(1)
        vec["bk_color"] = self.word_vec_list[idx][:,3,].unsqueeze(1)
        vec["height"] = self.height_list[idx]
        vec["coord"] = self.coord_list[idx]
        label = self.label_list[idx]
        word_list = self.all_word_list[idx]
        return vec, label, word_list


def make_word_img(word,font_size=24,hight=128,width=64,ttf='/workspace/dataset/OpenSans-Regular.ttf'):
    if len(word)>10:
        font_size = 14
    img_text = Image.new('RGB', (hight,width), (255,255,255))
    draw = ImageDraw.Draw(img_text)
    font = ImageFont.truetype(ttf, int(font_size))
    draw.text((hight//2,width//2), word, fill=(0),anchor="mm",font=font)
    img = np.array(img_text).astype('float32')
    img = img/255
    img = img.transpose(2,0,1)
    img = torch.from_numpy(img.astype(np.float32)).clone()
    img = img.unsqueeze(0)
    return img

def get_img(img_path,x_max=128,y_max=64):
    img = cv2.imread(img_path)
    h = img.shape[0]
    w = img.shape[1]
    img = cv2.resize(img,(x_max,(h*x_max)//w),cv2.INTER_LINEAR)
    h = img.shape[0]
    w = img.shape[1]
    # First resize to specified width
    
    if w > x_max and h > y_max: # Both width and height exceed specified size
        if w//h >= 2: # When aspect ratio is greater than 1:2
            img_r = cv2.resize(img,(x_max,(h*x_max)//w),cv2.INTER_LINEAR)
        elif w//h < 2:
            img_r = cv2.resize(img,((w*y_max)//h,y_max),cv2.INTER_LINEAR)
    elif w > x_max: # Only width exceeds specified size
        img_r = cv2.resize(img,(x_max,(x_max*h)//w),cv2.INTER_LINEAR)
    elif h > y_max: # For tall images, adjust height only
        img_r = cv2.resize(img,((w*y_max)//h,y_max),cv2.INTER_LINEAR)
    else: # Others remain as is
        img_r = img
    pad_x = (x_max - img_r.shape[1])//2
    pad_y = (y_max - img_r.shape[0])//2
    img_pad = cv2.copyMakeBorder(img_r, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, (0,0,0))
    if img_pad.shape[0] < y_max:
        img_pad = cv2.copyMakeBorder(img_pad, 1, 0, 0, 0, cv2.BORDER_CONSTANT, (0,0,0))
    if img_pad.shape[1] < x_max:
        img_pad = cv2.copyMakeBorder(img_pad, 0, 0, 1, 0, cv2.BORDER_CONSTANT, (0,0,0))
    img = img_pad/255
    img = img.transpose(2,0,1)
    img = torch.from_numpy(img.astype(np.float32)).clone()
    img = img.unsqueeze(0)
    return img

def get_color(img_path): 
    # rgb to L*a*b*
    img_ = cv2.imread(img_path)
    height,width,channel = np.shape(img_)

    # Get Lab (or RGB)
    img_Lab = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    img_L, img_a, img_b = cv2.split(img_Lab)
    img_L = img_L.astype(np.int16)*100/255
    img_a = img_a.astype(np.int16) - 128
    img_b = img_b.astype(np.int16) - 128
    
    # Otsu's binarization
    gray = cv2.cvtColor(img_, cv2.COLOR_RGB2GRAY)
    ret, img_o2 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    w,h = np.shape(img_o2)
    # Get pixels from peripheral area
    width_color = np.sum(img_o2[0,:])+np.sum(img_o2[w-1,:]) 
    hight_color = np.sum(img_o2[:,0])+np.sum(img_o2[:,h-1])
    color = width_color/255 + hight_color/255
    
    # Check peripheral and character area values, use larger one as background
    if color > w + h: # When peripheral color (number of 1s) exceeds half of peripheral pixels, periphery is background
        img_char = np.where(img_o2!=255,1,0)
        img_back = np.where(img_o2!=255,0,1)
    else :
        img_char = np.where(img_o2!=255,0,1)
        img_back = np.where(img_o2!=255,1,0)
    
    # Get average Lab of character area
    img_char_st = np.dstack((img_char,img_char))
    img_char_st = np.dstack((img_char_st,img_char))
    img_char_color = img_Lab * img_char_st
    img_char_color = img_char_color[np.all(img_char_color>0, axis=2)]
    # Count unique RGB elements and their frequencies
    value, count = np.unique((img_char_color.reshape(-1, img_char_color.shape[1])), axis=0, return_counts=True)
    # Get top 100 most frequent RGB colors
    idx = np.argsort(count)[::-1][0:100]
    if len(idx) == 0:
        pad_color = np.array([0,0,0])
    else:
        pad_color = value[idx[0]] # Color for padding
    pad_color = np.array([pad_color for i in range(100)])
    color = value[idx]
    color_num = color.shape[0]
    pad_color = pad_color[0:100-color_num]
    if len(pad_color) > 0:
        color = np.vstack((color,pad_color))
    char_color_tensor = torch.tensor(color).reshape(300)

    img_back_st = np.dstack((img_back,img_back))
    img_back_st = np.dstack((img_back_st,img_back))
    img_back_color = img_Lab * img_back_st
    img_back_color = img_back_color[np.all(img_back_color>0, axis=2)]
    value, count = np.unique((img_back_color.reshape(-1, img_back_color.shape[1])), axis=0, return_counts=True)
    idx = np.argsort(count)[::-1][0:100]
    if len(idx) == 0:
        pad_color = np.array([0,0,0])
    else:
        pad_color = value[idx[0]]
    pad_color = np.array([pad_color for i in range(100)])
    
    color = value[idx]
    color_num = color.shape[0]
    pad_color = pad_color[0:100-color_num]
    if len(pad_color) > 0:
        color = np.vstack((color,pad_color))
    back_color_tensor = torch.tensor(color).reshape(300)
    
    return char_color_tensor,back_color_tensor


def get_dataloader(batch_size,
                   data_type,
                   device = 'cuda',
                   token_num = 16, # Maximum number of words per book
                   csv="/workspace/dataset/AmazonBookCoverImages/WordList_and_BookCoverInfo.csv",
                   debug=True,
                   data_path='/workspace/dataset/AmazonBookCoverImages/genres/',
                   semantic_model_path = '/workspace/dataset/GoogleNews-vectors-negative300.bin.gz',
                   font_model_path = "/workspace/results/ResNet50_sw128_h64_lr0.001_v3/best_model.pth",
                   ext='.jpg'):

    with open('/workspace/dataset/AmazonBookCoverImages/csv/valid.pickle','rb') as f:
        book_cover_list_valid = pickle.load(f)

    model_semantic = gensim.models.KeyedVectors.load_word2vec_format(semantic_model_path, binary=True)
    model_font = ResNet50(class_num=2094,feat_dim=300).to(device)
    model_font.load_state_dict(torch.load(font_model_path,map_location=device))
    model_font.eval()

    if data_type == 'valid':
        df = pd.read_csv(f"/workspace/dataset/AmazonBookCoverImages/train.csv")
        df = df[df['split']=="train"].reset_index(drop=True)
        df = df[df['hight']>14]
        df = df[df['width']>14]
        df_org = pd.read_csv(f'/workspace/dataset/book30-listing-train.csv'\
                         ,encoding='cp932',names=("Amazon ID (ASIN)","Filename","Image URL","Title","Author","Category ID","Category"))
    else:
        df = pd.read_csv(f"/workspace/dataset/AmazonBookCoverImages/{data_type}.csv")
        df = df[df['split']==data_type].reset_index(drop=True)
        df = df[df['hight']>14]
        df = df[df['width']>14]
        df_org = pd.read_csv(f'/workspace/dataset/book30-listing-{data_type}.csv'\
                         ,encoding='cp932',names=("Amazon ID (ASIN)","Filename","Image URL","Title","Author","Category ID","Category"))
    df = pd.merge(df_org,df,on="Filename")
    df_tmp = df[~df['Filename'].duplicated()]
    book_cover_list = list(df_tmp["folder"].unique())
    if data_type == "train":
        for book in book_cover_list_valid:
            book_cover_list.remove(book)
    elif data_type == 'valid':
        book_cover_list = book_cover_list_valid
    # df = df[(df['word_num']>=3) & (df[df['order']>0])].reset_index(drop=True)
    # df = df[(df['word_num']>=3)].reset_index(drop=True)
    # df = df[(df['order']>0)].reset_index(drop=True)

    label_list = []
    all_vec_list = []
    all_title_list = []
    all_height_list = []
    all_coord_list = []
    zero_vec = torch.zeros((4,300)).to(device) # For semantic, font, ch_color, bk_color
    if debug == True:
        if data_type == 'train':
            book_cover_list = book_cover_list[0:20]
        else:
            book_cover_list = book_cover_list[0:5]

    
    for i in tqdm(range(len(book_cover_list))):
        # Temporary list for collecting information for one title
        vec_list = []
        height_list = []
        coord_list = []
        title_words = ''
        df_tmp = df[df['folder']==book_cover_list[i]].reset_index(drop=True)
        label = df_tmp.loc[0,'Category ID']
        category = df_tmp.loc[0,'Category']
        filename = df_tmp.loc[0,'Filename']
        foldar = 'res_'+filename.replace('.jpg','')

        for j in range(token_num): # Read title words one by one from target book cover
            if len(df_tmp)>j: # If word exists
                word = df_tmp['word'].values[j]
                title_words += word
                img_name = df_tmp['img_name'].values[j]
                img_path = '/workspace/dataset/word_detection_from_bookCover/dataset/'+category+'/word/' + foldar + '/'+ img_name
                if not os.path.isfile(img_path):
                    img_path = '/workspace/dataset/CannotRead/word/' + foldar + '/'+ img_name
                # Get semantic features
                w2v = model_semantic[word]
                semantic_vec = (torch.from_numpy(w2v.astype(np.float32)).clone()).to(device)    
                # Get font features
                try:
                    img = get_img(img_path)
                except:
                    print(img_path)
                _, fnot_vec = model_font(img.to(device))
                fnot_vec = fnot_vec.reshape(-1).detach()
                # Get color features
                ch_color,bk_color = get_color(img_path)
                height_list.append(torch.tensor([df_tmp.loc[j,"hight"]/df_tmp.loc[j,"book_cover_hight"]]))
                coord_list.append(torch.tensor([df_tmp.loc[j,"coord_x"]/df_tmp.loc[j,"book_cover_width"],\
                                                df_tmp.loc[j,"coord_y"]/df_tmp.loc[j,"book_cover_hight"]]))
                feat_vec = torch.stack([semantic_vec, fnot_vec, ch_color.to(device), bk_color.to(device)])
                vec_list.append(feat_vec)
                
                title_words += ' '
            else: # If no word, insert zero vector
                vec_list.append(zero_vec)
                height_list.append(torch.tensor([0]))
                coord_list.append(torch.tensor([0,0]))
                title_words += '* '

        vec_list = torch.stack(vec_list)
        all_vec_list.append(vec_list)

        height_list = torch.stack(height_list)
        all_height_list.append(height_list)
        coord_list = torch.stack(coord_list)
        all_coord_list.append(coord_list)

        all_title_list.append(title_words)
        label_list.append(label)
    trans1 = torchvision.transforms.ToTensor()
    dataset = Mydatasets(all_vec_list,label_list,all_height_list,all_coord_list,all_title_list,transform=trans1)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    return dataloader
