import torch
from models import VisionTransformer, CONFIGS, InputEmbed
import os
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataset_by_each_element import get_dataloader
import csv
MODEL = 'Transformer'
NUM_EPOCHS = 1000
BATCH_SIZE = 64
DEVICE = 'cuda'
EARLY_STOP = True
NUM_LAYER = 4
EARLY_STOP_NUM = 10
MODEL_SIZE = f"head6_layer{NUM_LAYER}"
MODEL_NAME = '1'
LR = 1e-5
SAVE_ROOT = "models"
SAVE_FOLDAR = f'{MODEL}_{MODEL_SIZE}_lr{LR}'
DROP_ELEMENTS = ["semantic"]
ELEMENTS = ["semantic","font_style","char_color","bk_color","height","coord"]
for dp_elemet in DROP_ELEMENTS:
    ELEMENTS.remove(dp_elemet)
    SAVE_FOLDAR += f'_{dp_elemet}_v2'
DEBUG = False

 
def torch_fix_seed(seed=0):
    print("used feat:",ELEMENTS)
    print("drop feat:",DROP_ELEMENTS)
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

def write_param():
    with open(f'/workspace/{SAVE_ROOT}/{SAVE_FOLDAR}/param.txt', mode='w') as f:
        f.write("MODEL:"+MODEL)
        f.write("NUM_EPOCHS:"+str(NUM_EPOCHS)+'\n')
        f.write("BATCH_SIZE:"+str(BATCH_SIZE)+'\n')
        f.write("DEVICE:"+DEVICE+'\n')
        f.write("EARLY_STOP:"+str(EARLY_STOP)+'\n')
        f.write("NUM_LAYER:"+str(NUM_LAYER)+'\n')
        f.write("EARLY_STOP_NUM:"+str(EARLY_STOP_NUM)+'\n')
        f.write("MODEL_SIZE:"+str(MODEL_SIZE)+'\n')
        f.write("MODEL_NAME:"+MODEL_NAME+'\n')
        f.write("SAVE_FOLDAR:"+SAVE_FOLDAR+'\n')
        drop_elements = ",".join(DROP_ELEMENTS)
        f.writelines("DROP_ELEMENTS:"+drop_elements+'\n')
        elements = ",".join(ELEMENTS)
        f.writelines("ELEMENTS:"+elements)

def train():
    print(SAVE_FOLDAR)
    torch_fix_seed(0)
    os.makedirs(f'/workspace/{SAVE_ROOT}/{SAVE_FOLDAR}',exist_ok=True)
    write_param()
    min_test_loss = np.inf 
    CONFIGS["title_level_config"]["transformer"]["num_layers"] = NUM_LAYER
    CONFIGS["word_level_config"]["transformer"]["token_num"] = 5
    title_level_config = CONFIGS["title_level_config"]
    word_level_config = CONFIGS["word_level_config"]
    model_title_level = VisionTransformer(title_level_config, num_classes=32, zero_head=False, vis=True).to(DEVICE)
    model_word_level = VisionTransformer(word_level_config, num_classes=32, zero_head=False, vis=True).to(DEVICE)
    model_word_level.head = nn.Identity()
    model_word_level.fc = nn.Identity()
    embed_xy = InputEmbed(input_size=2,output_size=300).to(DEVICE)
    embed_size = InputEmbed(input_size=1,output_size=300).to(DEVICE)

    cre_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam([*model_title_level.parameters(),*model_word_level.parameters(),
                            *embed_xy.parameters(),*embed_size.parameters()], lr=LR)

    train_loader = get_dataloader(batch_size=BATCH_SIZE,
                                  data_type='train',
                                  debug=DEBUG,
                                  token_num = 16
                                  )

    test_loader = get_dataloader(batch_size=BATCH_SIZE,
                                  data_type='valid',
                                  debug=DEBUG,
                                  token_num = 16
                                  )
                                
    with open(f'/workspace/{SAVE_ROOT}/{SAVE_FOLDAR}/loss.csv','w') as f:
        writer = csv.writer(f)
        writer.writerows([['epoch','train_loss','valid_loss','train_acc','test_acc']])
        for e in range(NUM_EPOCHS):
            train_loss = 0
            train_count = 0
            train_acc = 0
            model_title_level.train()
            model_word_level.train()
            for (data, labels, _) in train_loader:
                # data -> (batch, token_num, element_num, dim) element=(w2v,font,ch_color,bk_color)
                labels = labels.to(DEVICE)
                x_title = []
                x = None
                used_element = []
                if "semantic" in ELEMENTS:
                    x = data["semantic"].to(DEVICE,torch.float32)
                    used_element.append("semantic")
                if "font_style" in ELEMENTS:
                    used_element.append("font_style")
                    if x is None:
                        x = data["font_style"].to(DEVICE,torch.float32)
                    else:
                        x = torch.cat([x,data["font_style"].to(DEVICE,torch.float32)],dim=2)
                if "char_color" in ELEMENTS:
                    used_element.append("char_color")
                    if x is None:
                        x = data["char_color"].to(DEVICE,torch.float32)
                    else:
                        x = torch.cat([x,data["char_color"].to(DEVICE,torch.float32)],dim=2)
                if "bk_color" in ELEMENTS:
                    used_element.append("bk_color")
                    if x is None:
                        x = data["bk_color"].to(DEVICE,torch.float32)
                    else:
                        x = torch.cat([x,data["bk_color"].to(DEVICE,torch.float32)],dim=2)
                for i in range(x.shape[0]):
                    x_tmp = x[i]
                    if "height" in ELEMENTS:
                        if i==0:
                            used_element.append("height")
                        x_height = embed_size(data["height"].to(DEVICE,torch.float32)[i])
                        x_tmp = torch.cat([x_tmp,x_height.reshape(16,1,300)],dim=1)
                    if "coord" in ELEMENTS:
                        if i==0:
                            used_element.append("coord")
                        x_coord = embed_xy((data["coord"].to(DEVICE,torch.float32)[i]))
                        x_tmp = torch.cat([x_tmp,x_coord.reshape(16,1,300)],dim=1)
                    x_word,_ = model_word_level(x_tmp) # x[i] -> (token_num, element_num, dim), x_word -> (token_num,dim)
                    x_title.append(x_word)
                assert used_element == ELEMENTS, "not match between used feat. and selected feat."
                    
                x_title = torch.stack(x_title) # x_title -> (batch,token_num,dim)
                y,_ = model_title_level(x_title)
                loss = cre_loss(y, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * x.shape[0]
                train_count += x.shape[0]
                train_acc += torch.sum(torch.argmax(y,1) == labels).item()
            model_title_level.eval()
            model_word_level.eval()
            with torch.no_grad():
                test_loss = 0
                test_count = 0
                test_acc = 0
                for (data, labels, _) in test_loader:
                    # data -> (batch, token_num, element_num, dim) element=(w2v,font,ch_color,bk_color)
                    labels = labels.to(DEVICE)
                    x_title = []
                    x = None
                    if "semantic" in ELEMENTS:
                        x = data["semantic"].to(DEVICE,torch.float32)
                    if "font_style" in ELEMENTS:
                        if x is None:
                            x = data["font_style"].to(DEVICE,torch.float32) 
                        else:
                            x = torch.cat([x,data["font_style"].to(DEVICE,torch.float32)],dim=2)
                    if "char_color" in ELEMENTS:
                        if x is None:
                            x = data["char_color"].to(DEVICE,torch.float32)
                        else:
                            x = torch.cat([x,data["char_color"].to(DEVICE,torch.float32)],dim=2)
                    if "bk_color" in ELEMENTS:
                        if x is None:
                            x = data["bk_color"].to(DEVICE,torch.float32)
                        else:
                            x = torch.cat([x,data["bk_color"].to(DEVICE,torch.float32)],dim=2)
                    for i in range(x.shape[0]):
                        x_tmp = x[i]
                        if "height" in ELEMENTS:
                            x_height = embed_size(data["height"].to(DEVICE,torch.float32)[i])
                            x_tmp = torch.cat([x_tmp,x_height.reshape(16,1,300)],dim=1)
                        if "coord" in ELEMENTS:
                            x_coord = embed_xy((data["coord"].to(DEVICE,torch.float32)[i]))
                            x_tmp = torch.cat([x_tmp,x_coord.reshape(16,1,300)],dim=1)
                        x_word,_ = model_word_level(x_tmp) # x[i] -> (token_num, element_num, dim), x_word -> (token_num,dim)
                        x_title.append(x_word)

                    x_title = torch.stack(x_title) # x_title -> (batch,token_num,dim)
                    y,_ = model_title_level(x_title)
                    loss = cre_loss(y, labels)
                    test_loss += loss.item() * x.shape[0]
                    test_count += x.shape[0]
                    test_acc += torch.sum(torch.argmax(y,1) == labels).item()
            # Early stopping
            if test_loss/test_count <= min_test_loss:
                model_path = f'/workspace/{SAVE_ROOT}/{SAVE_FOLDAR}/best_model_title_level.pth'
                torch.save(model_title_level.state_dict(), model_path)
                model_path = f'/workspace/{SAVE_ROOT}/{SAVE_FOLDAR}/best_model_word_level.pth'
                torch.save(model_word_level.state_dict(), model_path)
                model_path = f'/workspace/{SAVE_ROOT}/{SAVE_FOLDAR}/best_model_size.pth'
                torch.save(embed_size.state_dict(), model_path)
                model_path = f'/workspace/{SAVE_ROOT}/{SAVE_FOLDAR}/best_model_coord.pth'
                torch.save(embed_xy.state_dict(), model_path)
                min_test_loss = test_loss/test_count
                count = 0
                update = 'Update best epoch {}'.format(e+1)
            elif (test_loss/test_count > min_test_loss) and (EARLY_STOP == True):
                count += 1
                update = ''
                if count == EARLY_STOP_NUM:
                    print('===early stop===')
                    break
            else:
                count = 0
            print(f'== {e+1} epoch == {update}')
            print('loss:')
            print(f'train:{train_loss/train_count}')
            print(f'valid:{test_loss/test_count}')
            print('accuracy:')
            print(f'train:{train_acc/train_count}')
            print(f'valid:{test_acc/test_count}')
            model_path = f'/workspace/{SAVE_ROOT}/{SAVE_FOLDAR}/least_model_title_level.pth'
            torch.save(model_title_level.state_dict(), model_path)
            model_path = f'/workspace/{SAVE_ROOT}/{SAVE_FOLDAR}/least_model_word_level.pth'
            torch.save(model_word_level.state_dict(), model_path)
            model_path = f'/workspace/{SAVE_ROOT}/{SAVE_FOLDAR}/least_model_size.pth'
            torch.save(embed_size.state_dict(), model_path)
            model_path = f'/workspace/{SAVE_ROOT}/{SAVE_FOLDAR}/least_model_coord.pth'
            torch.save(embed_xy.state_dict(), model_path)
            writer.writerows([[e+1,train_loss/train_count,test_loss/test_count,train_acc/train_count,test_acc/test_count]])


if __name__ == '__main__':
    train()
    print("used feat:",ELEMENTS)
    print("drop feat:",DROP_ELEMENTS)