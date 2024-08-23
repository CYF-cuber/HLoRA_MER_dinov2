from cgi import print_arguments
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.utils.data.dataloader as DataLoader
from torchsummary import summary
from sklearn.metrics import f1_score
from torchvision import transforms
import argparse
import os
from model import Fre_MER,Fre_MER_wo_phase_magnitude
from dinov2_mer import dinov2_lora_GAT_MER_3cls
from data_process import videoDataset

def train(args, model, train_dataset, test_dataset=None, train_log_file=None, test_log_file=None ,subject=""):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam([{"params":filter(lambda p: p.requires_grad, model.parameters())}], lr=args.lr, weight_decay=args.wdecay)
    '''
    optimizer = torch.optim.SGD([{"params":filter(lambda p: p.requires_grad, model.parameters())}], lr=args.lr, momentum=args.momentum,
                                weight_decay=args.wdecay, nesterov=True)
    '''
    train_dataloader = DataLoader.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    total_steps = 0
    keep_training = True
    epoch = 0
    final_acc = 0.0

    while keep_training:
        torch.cuda.empty_cache()
        epoch += 1
        args.test_mode = False
        totalsamples = 0
        correct_samples = 0
        acc = 0

        for i, item in enumerate(train_dataloader):
            print('-----epoch:{}  steps:{}/{}-----'.format(epoch, total_steps, args.num_steps))
            video, label = item
            print(video.size())
            #dy_loss = torch.zeros(1).requires_grad_(True)
            optimizer.zero_grad()
            pred_mer = model(video.to(device))
            pred_mer = F.log_softmax(pred_mer, dim=1)
            ME_loss = F.nll_loss(pred_mer, label.to(device))
            _, pred = torch.max(pred_mer, dim=1)

            print('label:{} pred:{}'.format(label, pred))
            print("ME_LOSS:",ME_loss )

            final_loss = ME_loss.to(torch.float32).cpu()* args.mer_weight

            final_loss.backward()
            optimizer.step()

            batch_correct_samples = pred.cpu().eq(label).sum()
            correct_samples += pred.cpu().eq(label).sum()
            totalsamples += len(label)
            batch_acc = batch_correct_samples / len(label)
            acc = correct_samples / totalsamples
            print("batch_acc:{}%".format(batch_acc * 100))
            print("acc:{}%".format(acc * 100))


            train_log_file.writelines('-----epoch:{}  steps:{}/{}-----\n'.format(epoch, total_steps, args.num_steps))
            train_log_file.writelines('final loss:{}'.format( final_loss))
            train_log_file.writelines('batch acc:{}\t\tacc:{}\n'.format(batch_acc * 100, acc * 100))
            total_steps += 1
            if total_steps == args.num_steps-1:
                #print(args.save_path+args.version+'{}.pth'.format(subject))
                torch.save(model.state_dict(), args.save_path+args.version+'_{}.pth'.format(subject))

            if total_steps > args.num_steps:
                keep_training = False
                break

        print("epoch average acc:{}%".format(acc * 100))
        print('=========================')
        train_log_file.writelines('epoch average acc:{}%\n'.format(acc * 100))
        train_log_file.writelines('=========================\n')
        acc = evaluate(args, model, epoch=epoch, test_dataset=test_dataset, test_log_file=test_log_file)
        if acc > final_acc:
            #torch.save(model.state_dict(), args.save_path)
            final_acc = acc

    return final_acc
    
def evaluate(args, model, epoch, test_dataset, test_log_file):

    args.test_mode = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    totalsamples = 0
    correct_samples = 0
    
    pred_list = []
    label_list = []

    global confusion_matrix 
    confusion_matrix = [[0,0,0],[0,0,0],[0,0,0]]


    test_dataloader = DataLoader.DataLoader(test_dataset, batch_size=48)

    total_loss = 0
    with torch.no_grad():
        for i, item in enumerate(test_dataloader):
            print('-----epoch:{}  batch:{}-----'.format(epoch, i))

            video, label = item
            pred_mer = model(video.to(device))

            pred_mer = F.log_softmax(pred_mer, dim=1)
            ME_loss = F.nll_loss(pred_mer, label.to(device))
            total_loss += ME_loss 
            _, pred = torch.max(pred_mer, dim=1)
            pred_list.extend(pred.cpu().numpy().tolist())

            label_list.extend(label.numpy().tolist())    

            print('label:{} \n pred:{}'.format(label, pred))

            correct_samples += cal_corr(label_list, pred_list)
            totalsamples += len(label_list)
            acc = correct_samples * 100.0 / totalsamples
            weighted_f1_score = f1_score(label_list, pred_list, average="weighted") * 100
            
        print('-----epoch:{}-----'.format(epoch))
        print("acc:{}%".format(acc))
        print("weighted f1 score:{}".format(weighted_f1_score))

        test_log_file.writelines('\n-----epoch:{}-----\n'.format(epoch))
        test_log_file.writelines('acc:{}\t\tweighted_f1:{}\n'.format(acc, weighted_f1_score))
        final_loss = total_loss.to(torch.float32).cpu()* args.mer_weight / totalsamples

        test_log_file.writelines('Final loss:{}\n'.format(final_loss))
        test_log_file.writelines('confusion_matrix:\n{}\n{}\n{}\n'.format(confusion_matrix[0],confusion_matrix[1],confusion_matrix[2]))
    
    print(confusion_matrix)
    return acc

def cal_corr(label_list, pred_list):
    corr = 0
    for (a, b) in zip(label_list, pred_list):
        confusion_matrix[a][b]+=1
        if a == b:
            corr += 1
    return corr


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_steps', type=int, default=600)
    parser.add_argument('--batch_size', type=int, default=36)
    parser.add_argument('--wdecay', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--mer_weight', type=float, default = 1)
    parser.add_argument('--dataset', type=str, default="SAMM")
    parser.add_argument('--save_path', default=None)
    parser.add_argument('--version', default='test')
    parser.add_argument('--net_test',action='store_true')
    parser.add_argument('--lora_depth', type=int, default=2)
    args = parser.parse_args()

    # if args.dataset == "CASME2":
    #     surprise_path = '../CASME2_data_5/surprise/'
    #     happiness_path = '../CASME2_data_5/happiness/'
    #     disgust_path = '../CASME2_data_5/disgust/'
    #     repression_path = '../CASME2_data_5/repression/'
    #     others_path = '../CASME2_data_5/others/'
    #     video_list = [surprise_path , happiness_path, disgust_path , repression_path , others_path]
    #     LOSO = ['17', '26', '16', '09', '05', '24', '02', '13', '04', '23', '11', '12', '08', '14', '03', '19', '01', '10', '20', '21', '22', '15', '06', '25', '07']

    # if args.dataset == "SAMM":
    #     surprise_path = '../SAMM_data_5/surprise/'
    #     happiness_path = '../SAMM_data_5/happiness/'
    #     anger_path = '../SAMM_data_5/anger/'
    #     contempt_path = '../SAMM_data_5/contempt/'
    #     others_path = '../SAMM_data_5/others/'
    #     video_list = [surprise_path , happiness_path, anger_path , contempt_path , others_path]
    #     #LOSO =['006','007','009','010','011','012','013','014','015','016','017','018','019','020','021','022','023','024','025','026','028','030','031','032','033','034','035','036','037']
    #     # del '024'
    #     LOSO =['006','007','009','010','011','012','013','014','015','016','017','018','019','020','021','022','023','025','026','028','030','031','032','033','034','035','036','037']

    '''##################################################################################################
    If you want to train 5-cls model, use Line 174-192 instead of 198-216 and change Line 108 to "    confusion_matrix = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]"
    '''
    
    if args.dataset == "CASME2":
        surprisepath_c = '../CASME2_data_3/surprise/'
        positivepath_c = '../CASME2_data_3/positive/'
        negativepath_c = '../CASME2_data_3/negative/'
        video_list = [surprisepath_c, positivepath_c, negativepath_c]
        subject_list = ['17', '26', '16', '09', '05', '24', '02', '13', '04', '23', '11', '12', '08', '14', '03', '19', '01', '10', '20', '21', '22', '15', '06', '25', '07']
        #[7, 8, 10, 12, 13, 17, 18, 19, 20, 24]
        LOSO = ['17', '26', '16', '09', '05', '24', '02', '13', '04', '23', '11', '12', '08', '14', '03', '19', '01', '10', '20', '21', '22', '15', '06', '25', '07']

        #LOSO = ['13', '14', '03']

    if args.dataset == "SAMM":
        surprisepath_s = '../SAMM_data_3/surprise/'
        positivepath_s = '../SAMM_data_3/positive/'
        negativepath_s = '../SAMM_data_3/negative/'
        video_list = [surprisepath_s, positivepath_s, negativepath_s]
        subject_list =['006','007','009','010','011','012','013','014','015','016','017','018','019','020','021','022','023','024','026','028','030','031','032','033','034','035','036','037']
        #[3, 6, 12,  14,  16, 17, 18, 19,  20,  21, 22,  24,  26,  27]
        LOSO =['006','007','009','010','011','012','013','014','015','016','017','018','019','020','021','022','023','024','026','028','030','031','032','033','034','035','036','037']

    test_log_file = open('../MultiTasks_log/' + args.version + '_test_log.txt', 'w')
    train_log_file = open('../MultiTasks_log/' + args.version + '_train_log.txt', 'w')

    train_log_file.writelines('----------args----------\n')
    test_log_file.writelines('----------args----------\n')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
        train_log_file.writelines('%s: %s\n' % (k, vars(args)[k]))
        test_log_file.writelines('%s: %s\n' % (k, vars(args)[k]))
    train_log_file.writelines('----------args----------\n')
    test_log_file.writelines('----------args----------\n')

    if args.save_path is None:
            args.save_path = './saved_models/'

    for sub in range(len(LOSO)):
        subject = LOSO[sub]
        test_subs = [subject]
        train_subs = []
        for i in subject_list:
            if i not in test_subs:
                train_subs.append(i)
        
        
        print('Loading test dataset '+ subject)
        test_dataset = torch.load('./mer_dataset/DINOV2_'+args.dataset+'_3cls_sub'+subject+'.pth')

        train_dataset = None
        for t in range(len(train_subs)):
            train_sub = train_subs[t]
            print('loading training dataset '+ train_sub)
            if t == 0:
                train_dataset =  torch.load('./mer_dataset/DINOV2_'+args.dataset+'_3cls_sub'+train_sub+'.pth')
                #print(train_dataset.__len__())
            else:            
                train_dataset = train_dataset + torch.load('./mer_dataset/DINOV2_'+args.dataset+'_3cls_sub'+train_sub+'.pth')
                #print(train_dataset.__len__())

        train_dataset_size = train_dataset.__len__()
        test_dataset_size = test_dataset.__len__()
        train_log_file.writelines('train_dataset_size:' + str(train_dataset_size))
        train_log_file.writelines('test_dataset_size:'+ str(test_dataset_size))
        print('train_dataset.size:{}'.format(len(train_dataset)))
        print('test_dataset.size:{}'.format(len(test_dataset)))
        train_log_file.writelines('LOSO ' +subject+'\n')
        test_log_file.writelines('LOSO '+subject+'\n')
        
        model = dinov2_lora_GAT_MER_3cls(args.lora_depth).cuda()
        final_acc=train(args=args, model=model, train_dataset=train_dataset, test_dataset=test_dataset,train_log_file=train_log_file, test_log_file=test_log_file,subject = subject)
        test_log_file.writelines('LOSO '+subject+' best_acc:'+str(final_acc)+'\n')
