import torch
from models import VisionTransformer, CONFIGS, AutoEncoder, ResNet50, InputEmbed
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataset import make_word_img, get_img, get_color
import csv
import pandas as pd
from tqdm import tqdm
import gensim
from visualization import visualize,get_word_img,get_visualization_res
from sklearn.metrics import confusion_matrix
import seaborn as sns
import shutil
import cv2
from PIL import Image
import pickle
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Book Cover Classification Testing')
    parser.add_argument('--model', type=str, default='Transformer', help='Model type')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--early_stop', action='store_true', default=True, help='Enable early stopping')
    parser.add_argument('--num_layer', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--early_stop_num', type=int, default=10, help='Early stop patience')
    parser.add_argument('--model_name', type=str, default='1', help='Model name')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--model_root', type=str, default='models', help='Model root directory')
    parser.add_argument('--save_root', type=str, default='./results', help='Save root directory')
    parser.add_argument('--drop_elements', nargs='*', default=[], help='Elements to drop')
    parser.add_argument('--elements', nargs='*', default=["semantic","font_style","char_color","bk_color","height","coord"], help='Elements to use')
    parser.add_argument('--token_num', type=int, default=16, help='Maximum number of tokens')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--font_model_path', type=str, default="/workspace/models/ResNet50_sw128_h64_lr0.001_v3/best_model.pth", help='Font model path')
    parser.add_argument('--semantic_model_path', type=str, default='/workspace/models/SemanticOnly_head6_layer4_lr1e-05_v1/best_model_semantic_only.pth', help='Semantic only model path')
    parser.add_argument('--word2vec_path', type=str, default='/workspace/dataset/GoogleNews-vectors-negative300.bin.gz', help='Word2Vec model path')
    parser.add_argument('--visualize', action='store_true', default=False, help='Generate visualization images')
    
    return parser.parse_args()

def torch_fix_seed(seed=0):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

def test(args):
    torch_fix_seed(args.seed)
    print("Test feat:", args.elements)
    print("Drop feat:", args.drop_elements)
    
    # Remove drop elements from elements list
    elements_copy = args.elements.copy()
    for drop_element in args.drop_elements:
        if drop_element in elements_copy:
            elements_copy.remove(drop_element)
    args.elements = elements_copy
    
    model_size = f"head6_layer{args.num_layer}"
    save_folder = f'{args.model}_{model_size}_lr{args.lr}'
    for drop_element in args.drop_elements:
        save_folder += f'_{drop_element}'

    CONFIGS["title_level_config"]["transformer"]["num_layers"] = args.num_layer
    title_level_config = CONFIGS["title_level_config"]
    word_level_config = CONFIGS["word_level_config"]
    model_title_level = VisionTransformer(title_level_config, num_classes=32, zero_head=False, vis=True).to(args.device)
    model_word_level = VisionTransformer(word_level_config, num_classes=32, zero_head=False, vis=True).to(args.device)
    model_word_level.head = nn.Identity()
    model_word_level.fc = nn.Identity()

    model_semantic = gensim.models.KeyedVectors.load_word2vec_format(args.word2vec_path, binary=True)

    embed_xy = InputEmbed(input_size=2,output_size=300).to(args.device)
    embed_size = InputEmbed(input_size=1,output_size=300).to(args.device)

    model_path = f'/workspace/{args.model_root}/{save_folder}/best_model_word_level.pth'
    model_word_level.load_state_dict(torch.load(model_path,map_location=args.device))

    model_path = f'/workspace/{args.model_root}/{save_folder}/best_model_title_level.pth'
    model_title_level.load_state_dict(torch.load(model_path,map_location=args.device))

    model_font = ResNet50(class_num=2094,feat_dim=300).to(args.device)
    model_font.load_state_dict(torch.load(args.font_model_path,map_location=args.device))

    model_path = f'/workspace/{args.model_root}/{save_folder}/best_model_coord.pth'
    embed_xy.load_state_dict(torch.load(model_path,map_location=args.device))

    model_path = f'/workspace/{args.model_root}/{save_folder}/best_model_size.pth'
    embed_size.load_state_dict(torch.load(model_path,map_location=args.device))

    df_test = pd.read_csv(f"/workspace/dataset/AmazonBookCoverImages/test.csv")
    df_test = df_test[df_test['split']=='test'].reset_index(drop=True)
    df_test = df_test[df_test['hight']>14]
    df_test = df_test[df_test['width']>14]
    df_test_org = pd.read_csv(f'/workspace/dataset/book30-listing-test.csv'
                    ,encoding='cp932',names=("Amazon ID (ASIN)","Filename","Image URL","Title","Author","Category ID","Category"))
    df_test = pd.merge(df_test_org,df_test,on="Filename")
    category_list = df_test["Category"].unique()

    token_num = args.token_num
    title_level_config = CONFIGS["title_level_config"]
    model_semantic_only = VisionTransformer(title_level_config, num_classes=32, zero_head=False, vis=True).to(args.device)
    model_semantic_only.load_state_dict(torch.load(args.semantic_model_path,map_location=args.device))

    model_word_level.eval()
    model_title_level.eval()
    model_font.eval()
    model_semantic_only.eval()

    df_res =  df_test[['folder','Title','Category','Category ID']].copy()
    df_res = df_res[~df_res.duplicated()].reset_index(drop=True)

    for i in range(5):
        k = str(i+1)
        col = f'top{k}_res'
        df_res[col] = 100
                    
    for category in tqdm(category_list):
        book_foldar_list = df_res[df_res['Category']==category]['folder'].values
        label_list = []
        all_vec_list = []
        all_title_list = []
        zero_vec = torch.zeros((len(args.elements),300)).to(args.device)
        attn_dict = {'foldar':[],'title':[],'elements':[],'word':[]}
        with torch.no_grad():
            for book_foldar in book_foldar_list:
                word_vec_list = []
                vec_list = []
                word_list = ''
                df_tmp = df_test[df_test['folder']==book_foldar].reset_index(drop=True)
                label = df_tmp.loc[0,'Category ID']
                category = df_tmp.loc[0,'Category']
                filename = df_tmp.loc[0,'Filename']
                foldar = book_foldar
                for j in range(token_num):
                    if len(df_tmp)>j:
                        # Get semantic features
                        word = df_tmp['word'].values[j]
                        word_list += word
                        w2v = model_semantic[word]
                        semantic_vec = (torch.from_numpy(w2v.astype(np.float32)).clone()).to(args.device)
                        w2v_torch = (torch.from_numpy(w2v.astype(np.float32)).clone()).to(args.device)
                        word_vec_list.append(w2v_torch)
                        # Get font features
                        img_name = df_tmp['img_name'].values[j]
                        img_path = '/workspace/dataset/word_detection_from_bookCover/dataset/'+category+'/word/' + foldar + '/'+ img_name
                        if not os.path.isfile(img_path):
                            img_path = '/workspace/dataset/CannotRead/word/' + foldar + '/'+ img_name
                        img = get_img(img_path) 
                        _, font_vec = model_font(img.to(args.device))
                        font_vec = font_vec.reshape(-1).detach()
                        # Get color features
                        ch_color, bk_color = get_color(img_path)

                        # Get height
                        height = torch.tensor([df_tmp.loc[j,"hight"]/df_tmp.loc[j,"book_cover_hight"]])
                        x_height = embed_size(height.to(args.device,torch.float32))

                        # Get coordinates
                        coord = torch.tensor([df_tmp.loc[j,"coord_x"]/df_tmp.loc[j,"book_cover_width"],df_tmp.loc[j,"coord_y"]/df_tmp.loc[j,"book_cover_hight"]])
                        x_coord = embed_xy(coord.to(args.device,torch.float32))
                        
                        data_list = []
                        if "semantic" in args.elements:
                            data_list.append(semantic_vec)
                        if "font_style" in args.elements:
                            data_list.append(font_vec)
                        if "char_color" in args.elements:
                            data_list.append(ch_color.to(args.device))
                        if "bk_color" in args.elements:
                            data_list.append(bk_color.to(args.device))
                        if "height" in args.elements:
                            data_list.append(x_height)
                        if "coord" in args.elements:
                            data_list.append(x_coord)
                        feat_vec = torch.stack(data_list)
                        vec_list.append(feat_vec)
                        word_list += ' '
                        
                    else:
                        word_vec_list.append(torch.zeros((300)).to(args.device))
                        vec_list.append(zero_vec)
                        word_list += '* '
                        
                vec_list = torch.stack(vec_list)
                word_vec_list = torch.stack(word_vec_list)
                x_word,_ = model_word_level(vec_list) 
                y,_ = model_title_level(x_word.reshape(1,token_num,300))
                for i in range(5):
                    topk_res = torch.argsort(y,descending=True)[0,i].item()
                    k = str(i+1)
                    col = f'top{k}_res'
                    df_res.loc[df_res['folder']==foldar,col] = int(topk_res)

                attn,logits = visualize(x_word.reshape(1,token_num,300),model_title_level.to(args.device),device=args.device)
                attn_element,logits_element = visualize(vec_list.reshape(token_num,len(args.elements),300),model_word_level.to(args.device),device=args.device)
                mask_semantic,_ = visualize(word_vec_list.reshape(1,token_num,300),model_semantic_only.to(args.device),device=args.device)
                
                # Generate visualization images if requested
                if args.visualize:
                    img_all = 0
                    idx = 0
                    words = word_list.split(' ')
                    for i in range(len(words)):
                        word = words[i]
                        if word == '':
                            break
                        else:
                            rate = attn[:,i].item()
                            rate_design = attn_element[i,:].cpu().tolist()
                            rate_semantic = mask_semantic[:,i].item()
                            img = get_visualization_res(word,rate,rate_design,width=60,height=30,element_num=len(args.elements))
                            w,h,_ = img.shape
                            img_semantic = get_word_img(word,rate_semantic,width=h,height=w)
                            if i == 0:  
                                img_all = img
                                img_semantic_all = img_semantic
                            else:
                                img_all = np.hstack((img_all,img))
                                img_semantic_all = np.hstack((img_semantic_all,img_semantic))
                    img_all = np.vstack((img_all, img_semantic_all))
                else:
                    words = word_list.split(' ')
                # Save book cover images
                book_id = foldar.split("res_")[-1] if "res_" in foldar else foldar
                book_path = f'/workspace/dataset/AmazonBookCoverImages/genres/{category}/{book_id}.jpg'
                detect_book_path = f'/workspace/dataset/word_detection_from_bookCover/dataset/{category}/ditection/{foldar}.jpg'
                if not os.path.isfile(book_path):
                    book_path = f'/workspace/dataset/AmazonBookCoverImages/CannotRead/{book_id}.jpg'
                    detect_book_path = f'/workspace/dataset/CannotRead/ditection/{foldar}.jpg'
                    
                # Create save directories and copy images
                save_folder = f'{args.model}_head6_layer{args.num_layer}_lr{args.lr}'
                for drop_element in args.drop_elements:
                    save_folder += f'_{drop_element}'
                    
                os.makedirs(f'{args.save_root}/bookcover/{category}/', exist_ok=True)
                save_book_path = f'{args.save_root}/bookcover/{category}/{foldar}.png'
                if os.path.isfile(book_path):
                    shutil.copy(book_path, save_book_path)
                    
                os.makedirs(f'/workspace/{args.save_root}/detect_bookcover/{category}/', exist_ok=True)
                save_detect_book_path = f'/workspace/{args.save_root}/detect_bookcover/{category}/{foldar}.png'
                if os.path.isfile(detect_book_path):
                    shutil.copy(detect_book_path, save_detect_book_path)

                attn_dict['foldar'].append(foldar)
                attn_dict['title'].append(words)
                attn_dict['elements'].append(attn_element.to('cpu').detach().numpy().copy())
                attn_dict['word'].append(attn.to('cpu').detach().numpy().copy())

                # Save visualization if generated
                if args.visualize:
                    fig = plt.figure(figsize=(15,5))
                    fig.tight_layout()
                    plt.imshow(img_all)
                    plt.axis('off')
                    os.makedirs(f'/workspace/{args.save_root}/visualization_res/{save_folder}/{category}/', exist_ok=True)
                    plt.savefig(f'/workspace/{args.save_root}/visualization_res/{save_folder}/{category}/{foldar}.png', bbox_inches="tight")
                    plt.close()

        # Save attention data for each category
        os.makedirs(f'/workspace/{args.save_root}/attention/{save_folder}/', exist_ok=True)
        with open(f'/workspace/{args.save_root}/attention/{save_folder}/{category}.pickle', "wb") as f:
            pickle.dump(attn_dict, f)
    
    # Create zip archives and save results
    if args.visualize:
        shutil.make_archive(f'/workspace/{args.save_root}/visualization_res', 'zip', root_dir=f'/workspace/{args.save_root}/visualization_res')
    shutil.make_archive(f'/workspace/{args.save_root}/bookcover', 'zip', root_dir=f'/workspace/{args.save_root}/bookcover')
    
    # Save results CSV
    name = ""
    for drop_element in args.drop_elements:
        name += drop_element
    df_res.to_csv(f'/workspace/{args.save_root}/csv/{name}_res.csv')
    
    print("Test completed successfully!")
    return df_res, attn_dict

if __name__ == '__main__':
    args = parse_args()
    df_results, attention_dict = test(args)