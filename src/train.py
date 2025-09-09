import torch
from models import VisionTransformer, CONFIGS, InputEmbed
import os
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataset_by_each_element import get_dataloader
import csv
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Book Cover Classification Training')
    parser.add_argument('--model', type=str, default='Transformer', help='Model type')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--early_stop', action='store_true', default=True, help='Enable early stopping')
    parser.add_argument('--num_layer', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--early_stop_num', type=int, default=10, help='Early stop patience')
    parser.add_argument('--model_name', type=str, default='1', help='Model name')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--save_root', type=str, default='models', help='Save root directory')
    parser.add_argument('--drop_elements', nargs='*', default=[], help='Elements to drop')
    parser.add_argument('--elements', nargs='*', default=["semantic","font_style","char_color","bk_color","height","coord"], help='Elements to use')
    parser.add_argument('--token_num', type=int, default=16, help='Maximum number of tokens')
    parser.add_argument('--debug', action='store_true', default=False, help='Enable debug mode')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--amazon_csv_path', type=str, default='/workspace/dataset/AmazonBookCoverImages', help='Path to Amazon CSV files')
    parser.add_argument('--book_listing_csv_path', type=str, default='/workspace/dataset', help='Path to book listing CSV files')
    parser.add_argument('--valid_pickle_path', type=str, default='/workspace/dataset/AmazonBookCoverImages/csv/valid.pickle', help='Path to validation pickle file')
    
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

def write_param(args, save_folder):
    with open(f'/workspace/{args.save_root}/{save_folder}/param.txt', mode='w') as f:
        f.write("MODEL:"+args.model+'\n')
        f.write("NUM_EPOCHS:"+str(args.num_epochs)+'\n')
        f.write("BATCH_SIZE:"+str(args.batch_size)+'\n')
        f.write("DEVICE:"+args.device+'\n')
        f.write("EARLY_STOP:"+str(args.early_stop)+'\n')
        f.write("NUM_LAYER:"+str(args.num_layer)+'\n')
        f.write("EARLY_STOP_NUM:"+str(args.early_stop_num)+'\n')
        f.write("MODEL_NAME:"+args.model_name+'\n')
        f.write("SAVE_FOLDER:"+save_folder+'\n')
        drop_elements = ",".join(args.drop_elements)
        f.writelines("DROP_ELEMENTS:"+drop_elements+'\n')
        elements = ",".join(args.elements)
        f.writelines("ELEMENTS:"+elements)

def train(args):
    torch_fix_seed(args.seed)
    print("used feat:", args.elements)
    print("drop feat:", args.drop_elements)
    
    # Remove drop elements from elements list
    elements_copy = args.elements.copy()
    for drop_element in args.drop_elements:
        if drop_element in elements_copy:
            elements_copy.remove(drop_element)
    args.elements = elements_copy
    
    model_size = f"head6_layer{args.num_layer}"
    save_folder = f'{args.model}_{model_size}_lr{args.lr}'
    for drop_element in args.drop_elements:
        save_folder += f'_{drop_element}_v2'
    
    print(save_folder)
    os.makedirs(f'/workspace/{args.save_root}/{save_folder}', exist_ok=True)
    write_param(args, save_folder)
    
    min_test_loss = np.inf 
    CONFIGS["title_level_config"]["transformer"]["num_layers"] = args.num_layer
    CONFIGS["word_level_config"]["transformer"]["token_num"] = 5
    title_level_config = CONFIGS["title_level_config"]
    word_level_config = CONFIGS["word_level_config"]
    model_title_level = VisionTransformer(title_level_config, num_classes=32, zero_head=False, vis=True).to(args.device)
    model_word_level = VisionTransformer(word_level_config, num_classes=32, zero_head=False, vis=True).to(args.device)
    model_word_level.head = nn.Identity()
    model_word_level.fc = nn.Identity()
    embed_xy = InputEmbed(input_size=2, output_size=300).to(args.device)
    embed_size = InputEmbed(input_size=1, output_size=300).to(args.device)

    cre_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam([*model_title_level.parameters(), *model_word_level.parameters(),
                            *embed_xy.parameters(), *embed_size.parameters()], lr=args.lr)

    train_loader = get_dataloader(batch_size=args.batch_size,
                                  data_type='train',
                                  debug=args.debug,
                                  token_num=args.token_num,
                                  amazon_csv_path=args.amazon_csv_path,
                                  book_listing_csv_path=args.book_listing_csv_path,
                                  valid_pickle_path=args.valid_pickle_path)

    test_loader = get_dataloader(batch_size=args.batch_size,
                                 data_type='valid',
                                 debug=args.debug,
                                 token_num=args.token_num,
                                 amazon_csv_path=args.amazon_csv_path,
                                 book_listing_csv_path=args.book_listing_csv_path,
                                 valid_pickle_path=args.valid_pickle_path)
                                
    with open(f'/workspace/{args.save_root}/{save_folder}/loss.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows([['epoch', 'train_loss', 'valid_loss', 'train_acc', 'test_acc']])
        for e in range(args.num_epochs):
            train_loss = 0
            train_count = 0
            train_acc = 0
            model_title_level.train()
            model_word_level.train()
            for (data, labels, _) in train_loader:
                # data -> (batch, token_num, element_num, dim) element=(w2v,font,ch_color,bk_color)
                labels = labels.to(args.device)
                x_title = []
                x = None
                used_element = []
                if "semantic" in args.elements:
                    x = data["semantic"].to(args.device, torch.float32)
                    used_element.append("semantic")
                if "font_style" in args.elements:
                    used_element.append("font_style")
                    if x is None:
                        x = data["font_style"].to(args.device, torch.float32)
                    else:
                        x = torch.cat([x, data["font_style"].to(args.device, torch.float32)], dim=2)
                if "char_color" in args.elements:
                    used_element.append("char_color")
                    if x is None:
                        x = data["char_color"].to(args.device, torch.float32)
                    else:
                        x = torch.cat([x, data["char_color"].to(args.device, torch.float32)], dim=2)
                if "bk_color" in args.elements:
                    used_element.append("bk_color")
                    if x is None:
                        x = data["bk_color"].to(args.device, torch.float32)
                    else:
                        x = torch.cat([x, data["bk_color"].to(args.device, torch.float32)], dim=2)
                for i in range(x.shape[0]):
                    x_tmp = x[i]
                    if "height" in args.elements:
                        if i == 0:
                            used_element.append("height")
                        x_height = embed_size(data["height"].to(args.device, torch.float32)[i])
                        x_tmp = torch.cat([x_tmp, x_height.reshape(args.token_num, 1, 300)], dim=1)
                    if "coord" in args.elements:
                        if i == 0:
                            used_element.append("coord")
                        x_coord = embed_xy((data["coord"].to(args.device, torch.float32)[i]))
                        x_tmp = torch.cat([x_tmp, x_coord.reshape(args.token_num, 1, 300)], dim=1)
                    x_word, _ = model_word_level(x_tmp)  # x[i] -> (token_num, element_num, dim), x_word -> (token_num,dim)
                    x_title.append(x_word)
                assert used_element == args.elements, "not match between used feat. and selected feat."
                    
                x_title = torch.stack(x_title)  # x_title -> (batch,token_num,dim)
                y, _ = model_title_level(x_title)
                loss = cre_loss(y, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * x.shape[0]
                train_count += x.shape[0]
                train_acc += torch.sum(torch.argmax(y, 1) == labels).item()
            model_title_level.eval()
            model_word_level.eval()
            with torch.no_grad():
                test_loss = 0
                test_count = 0
                test_acc = 0
                for (data, labels, _) in test_loader:
                    # data -> (batch, token_num, element_num, dim) element=(w2v,font,ch_color,bk_color)
                    labels = labels.to(args.device)
                    x_title = []
                    x = None
                    if "semantic" in args.elements:
                        x = data["semantic"].to(args.device, torch.float32)
                    if "font_style" in args.elements:
                        if x is None:
                            x = data["font_style"].to(args.device, torch.float32) 
                        else:
                            x = torch.cat([x, data["font_style"].to(args.device, torch.float32)], dim=2)
                    if "char_color" in args.elements:
                        if x is None:
                            x = data["char_color"].to(args.device, torch.float32)
                        else:
                            x = torch.cat([x, data["char_color"].to(args.device, torch.float32)], dim=2)
                    if "bk_color" in args.elements:
                        if x is None:
                            x = data["bk_color"].to(args.device, torch.float32)
                        else:
                            x = torch.cat([x, data["bk_color"].to(args.device, torch.float32)], dim=2)
                    for i in range(x.shape[0]):
                        x_tmp = x[i]
                        if "height" in args.elements:
                            x_height = embed_size(data["height"].to(args.device, torch.float32)[i])
                            x_tmp = torch.cat([x_tmp, x_height.reshape(args.token_num, 1, 300)], dim=1)
                        if "coord" in args.elements:
                            x_coord = embed_xy((data["coord"].to(args.device, torch.float32)[i]))
                            x_tmp = torch.cat([x_tmp, x_coord.reshape(args.token_num, 1, 300)], dim=1)
                        x_word, _ = model_word_level(x_tmp)  # x[i] -> (token_num, element_num, dim), x_word -> (token_num,dim)
                        x_title.append(x_word)

                    x_title = torch.stack(x_title)  # x_title -> (batch,token_num,dim)
                    y, _ = model_title_level(x_title)
                    loss = cre_loss(y, labels)
                    test_loss += loss.item() * x.shape[0]
                    test_count += x.shape[0]
                    test_acc += torch.sum(torch.argmax(y, 1) == labels).item()

            train_loss = train_loss / train_count
            train_acc = train_acc / train_count
            test_loss = test_loss / test_count
            test_acc = test_acc / test_count
            print(f'Epoch: {e+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
            writer.writerow([e+1, train_loss, test_loss, train_acc, test_acc])

            # Early stopping
            if test_loss < min_test_loss:
                early_stop_count = 0
                min_test_loss = test_loss
                torch.save(model_title_level.state_dict(), f'/workspace/{args.save_root}/{save_folder}/best_model_title_level.pth')
                torch.save(model_word_level.state_dict(), f'/workspace/{args.save_root}/{save_folder}/best_model_word_level.pth')
                torch.save(embed_xy.state_dict(), f'/workspace/{args.save_root}/{save_folder}/best_model_coord.pth')
                torch.save(embed_size.state_dict(), f'/workspace/{args.save_root}/{save_folder}/best_model_size.pth')
            else:
                early_stop_count += 1

            if args.early_stop and early_stop_count >= args.early_stop_num:
                print(f'Early stopping at epoch {e+1}')
                break

if __name__ == '__main__':
    args = parse_args()
    train(args)