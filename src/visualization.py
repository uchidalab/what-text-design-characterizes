# import sys
# sys.path.append('/workspace/vit_otao/')
import typing
import io
import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from urllib.request import urlretrieve
from torchvision import transforms
from models import VisionTransformer, CONFIGS
from PIL import Image, ImageDraw, ImageFont
import matplotlib

def visualize(data,model,device):
    eps = 10e-10
    model.eval()
    with torch.no_grad():
        logits, att_mat = model(data)
    att_mat = torch.stack(att_mat)#.squeeze(1)
    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=2)
    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(-1)).to(device)
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size()).to(device)
    joint_attentions[0] = aug_att_mat[0]
    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    # Attention from the output token to the input space.
    mask = v[:, 0, 1:]
    mask_max = torch.max(mask.reshape(mask.shape[0],-1),1).values
    mask_max = mask_max.reshape(mask.shape[0],1)
    mask_min = torch.min(mask.reshape(mask.shape[0],-1),1).values
    mask_min = mask_min.reshape(mask.shape[0],1)
    mask = (mask-mask_min)/(mask_max-mask_min+eps)
    return mask, logits



def get_word_img(word,rate,font_size=35,height=180,width=120,ttf='/workspace/dataset/OpenSans-Regular.ttf'):
    cmap = matplotlib.pyplot.get_cmap('jet')
    if len(word)>10:
        font_size = 25
    img_text = Image.new('RGB', (width,height), (255,255-int(255*rate),255-int(255*rate))) #(400,200)
    draw = ImageDraw.Draw(img_text)
    try:
        font = ImageFont.truetype(ttf, int(font_size))
    except OSError:
        # Fallback to default font if custom font is not available
        font = ImageFont.load_default()
    draw.text((width//2,height//2), word, fill=(0,0,0),anchor="mm",font=font)
    # draw.rectangle([(0, 0), (width-1,height-1)], outline=(0,0,0),width=1)
    draw.line((width-1,0,width-1,height-1),fill=(0,0,0), width=1)
    draw.line((0,0,0,height-1),fill=(0,0,0), width=1)
    draw.line((0,height-1,width-1,height-1),fill=(0,0,0), width=1)
    img = np.array(img_text).astype('uint8')
    return img

def get_visualization_res(word,rate,rate_design,font_size=35,width=180,height=120,ttf='/workspace/dataset/OpenSans-Regular.ttf',element_num=6):
    cmap = matplotlib.pyplot.get_cmap('jet')
    if len(word)>10:
        font_size = 25
    # 単語の描画
    img_text = Image.new('RGB', (width,height), (255,255-int(255*rate),255-int(255*rate))) #(400,200)
    draw = ImageDraw.Draw(img_text)
    try:
        font = ImageFont.truetype(ttf, int(font_size))
    except OSError:
        # Fallback to default font if custom font is not available
        font = ImageFont.load_default()
    draw.text((width//2,height//2), word, fill=(0,0,0),anchor="mm",font=font)
    draw.line((width-1,0,width-1,height-1),fill=(0,0,0), width=1)
    draw.line((0,0,0,height-1),fill=(0,0,0), width=1)
#     draw.line((0,height-1,width-1,height-1),fill=(0,0,0), width=1)
    img = np.array(img_text).astype('uint8')
    # attentionの描画
    accumulated_height = 0
    for i in range(element_num):
        # Calculate height for this element
        if i == element_num - 1:
            # Last element gets remaining height to avoid rounding errors
            elem_height = height - accumulated_height
        else:
            elem_height = height // element_num
        
        img_text = Image.new('RGB', (width//element_num, elem_height), (255,255-int(255*rate_design[i]),255-int(255*rate_design[i])))
        draw = ImageDraw.Draw(img_text)
        draw.rectangle([(0, 0), (width//element_num-1, elem_height-1)], fill=(255,255-int(255*rate_design[i]),255-int(255*rate_design[i])), outline=(0,0,0),width=1)
        
        if i == 0:
            img_tmp = np.array(img_text).astype('uint8')
        else:
            im = np.array(img_text).astype('uint8')
            img_tmp = np.vstack((img_tmp,im))
        
        accumulated_height += elem_height
    img = np.hstack((img,img_tmp))
    return img