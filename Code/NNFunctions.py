import torch
from torch.autograd import Variable
import numpy as np
from scipy.special import comb
from Parse_args import *
import copy
import random
from sklearn.decomposition import PCA

FC_Dim = opt.ROI_num
def corr_loss(x, y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    cost = (torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))))
    return cost

def CompressImg(img):
    #print('img.cpu().detach().numpy() shape {0}, type {1}'.format(img.cpu().detach().numpy().shape, type(img.cpu().detach().numpy())))
    pca = PCA(n_components = 301).fit(img.cpu().detach().numpy())   # img 360 * 360
    com_img = pca.transform( np.transpose(pca.transform(img.cpu().detach().numpy()) ) ) 
    #print('com_img shape and type {0} and {1}'.format(com_img.shape, type(com_img)))
    img_recon = pca.inverse_transform( np.transpose(pca.inverse_transform(com_img)) )
    PrintError(np.array(img.cpu().detach().numpy(), dtype='double'), np.array(img_recon, dtype='double'))
    #print('com img shape {0}, type {1}'.format(torch.from_numpy(com_img).to(device).size(), type(torch.from_numpy(com_img).to(device))))
    return torch.from_numpy(com_img)
    

def PrintError(data, recdata):
    sum1 = 0
    sum2 = 0
    D_value = data - recdata # 计算两幅图像之间的差值矩阵
    # 计算两幅图像之间的误差率，即信息丢失率
    for i in range(data.shape[0]):
        sum1 += np.dot(data[i],data[i])
        sum2 += np.dot(D_value[i], D_value[i])
    #print('Lost Information：', sum2)
    #print('Original Information：', sum1)
    #print('Lost Information Ratio：', sum2/sum1)

def AgeGroupOneHotEncoding(age_groups):
    #OHElength = len(opt.age_group)
    #print('OHElength ', OHElength)
    #OHEAgeGroups = torch.eye(OHElength)[age_groups.squeeze().long()]
    #print('OHEAgeGroups.size() ', list(OHEAgeGroups.size()))
    #OHEAgeGroups = OHEAgeGroups.unsqueeze(-1).unsqueeze(-1).repeat(1,1,360, 360)
    #print('OHEAgeGroups size is ', OHEAgeGroups.size())
    OHEAgeGroups = torch.zeros(len(age_groups),len(opt.age_group))
    for i in range(len(age_groups)):
         OHEAgeGroups[i, age_groups[i].long()] = 1
    return OHEAgeGroups.to(device)

def FCRecovery(recon_test, mode = 'num'):
    if mode == 'num':
        FC_recon = np.zeros((FC_Dim,FC_Dim))
        for i in range(FC_Dim): #360
            for j in range(FC_Dim):
                if(i == j):
                    FC_recon[i,j] = 1
                elif(j > i):
                    FC_recon[i,j] = recon_test[0, FC_Dim * i + j - int((i+1)*(i+2)/2)]
        temp = np.triu(FC_recon, 1)
        FC_recon = FC_recon + np.transpose(temp)
    elif mode == 'tensor':
        FC_recon = torch.zeros((len(recon_test),FC_Dim,FC_Dim))
        for k in range(len(recon_test)):
            temp = torch.zeros(FC_Dim,FC_Dim)
            for i in range(FC_Dim): #360ROIS
                for j in range(i+1,FC_Dim,1):
                    if(j > i):
                        #temp[i,j] = recon_test[k, 360 * i + j - int((i+1)*(i+2)/2)]
                        FC_recon[k,i,j] = recon_test[k, FC_Dim * i + j - int((i+1)*(i+2)/2)]
            FC_recon[k,:,:] += torch.eye(FC_Dim)
            FC_recon[k,:,:] += torch.transpose(torch.triu(FC_recon[k,:,:], 1),0,1)  #up triangle
            #print('temp type ',type(temp))
            #print('Enter CompressImg')
            if torch.max(torch.isnan(temp)):
               input('NaN occurs in temp')
            #FC_recon[k,:,:] = CompressImg( temp + torch.transpose(temp_UT,0,1))
    return FC_recon


def StrProcess(lines):
    dataset = []
    for line in lines:
        line = line.strip('\n')
        dataset.append(str(line))
        #print((dataset))
    return dataset

def findMultiTimeSubinTeRealIDs(te_realIDs):
    f = open(opt.MultiTimeSubFile,"r")
    lines = f.readlines()
    f.close()
    lines = StrProcess(lines)
    multi_time_subs = []
    multi_time_subs_pos = []
    for line in lines:
        if(line in te_realIDs):
            multi_time_subs.append(line)
            multi_time_subs_pos.append(te_realIDs.index(line))
    return multi_time_subs,multi_time_subs_pos

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
        torch.nn.init.constant_(m.bias.data, 0.1)

def reparameterization(mu, logvar):      
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), opt.noiseSize))))   
    z = sampled_z * std.to(device) + mu.to(device)

    return z

def corr(Spec_s, Spec_f):
        out = torch.zeros((Spec_f.shape[0],1))
        for i in range(Spec_f.shape[0]):
            vx = Spec_s[i,:] - torch.mean(Spec_s[i,:])
            vy = Spec_f[i,:] - torch.mean(Spec_f[i,:])
            out[i,0]=(torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))))
        return out

def Triplet_corr(pair1, pair2, nega):
    relation_pari1_nega = corr(pair1, nega)
    relation_pair2_nega = corr(pair2, nega)
    relation_pair1_pair2 = corr(pair1, pair2) 
    return relation_pair1_pair2, relation_pari1_nega, relation_pair2_nega

def TripleLoss(relation_pair1_pair2, relation_pari1_nega, relation_pair2_nega):
    return (-2 * relation_pair1_pair2 ).sum()  #+ relation_pari1_nega + relation_pair2_nega

def AgeLoss(RealAge, EstiAge):
    loss_sum = abs(RealAge - EstiAge).sum()
    return loss_sum/len(RealAge)


def findIDPos(ID, IDs):
    location = 0
    pos = []
    for _ in range(IDs.count(ID)):
        location += IDs[location:].index(ID)    
        pos.append( location )
        location += 1
    #print(ID, pos)
    return pos


def GeneTriplet(tr_FCs, tr_RealIDs, tr_Days):   # two same subject and one another subject to generate triplet
    #pair_days = []
    set_tr_realID = set(tr_RealIDs)
    TripletList = []
    #TripletList structure is: elements in Triplet is dict, and each dict key indicates the order of triplet, values includes two same subjects and one negative subject
    cnt = 0
    for realID in set_tr_realID:  
        pos = findIDPos(realID, tr_RealIDs) 
        if (len(pos) > 1):
            comb_Num = int(comb(len(pos),2))                 # how many the specific subject (at least 2 times) can combine 
            for j in range(len(pos)):
                fir = pos[j]
                for k in range(j+1, len(pos)):
                    sec = pos[k]
                    nega_set_realID = copy.deepcopy(set_tr_realID)
                    nega_set_realID.remove(realID)
                    for nega_realID in nega_set_realID:
                        nega_pos = findIDPos(nega_realID, tr_RealIDs)
                        for i in range(len(nega_pos)):
                            tempDict = {}
                            templist = []
                            templist.append(tr_FCs[fir])
                            templist.append(tr_Days[fir])
                            templist.append(tr_FCs[sec])
                            templist.append(tr_Days[sec])
                            templist.append(tr_FCs[nega_pos[i]])
                            templist.append(tr_Days[nega_pos[i]])
                            tempDict.setdefault(cnt, templist)
                            TripletList.append(tempDict)
                            cnt += 1
                            #pair_days.append(tr_Days[fir])
                            #pair_days.append(tr_Days[sec])
                            #break
                    #break
                #break
            #break
    #print('pair_days are', set(pair_days))
    random.shuffle(TripletList)
    return TripletList, cnt

def DisTriplet(OneTriplet):
    key = list(OneTriplet.keys())[0]
    values = OneTriplet[key]
    pair1_FC = torch.Tensor(values[0]).reshape(1,-1)
    pair1_age = torch.Tensor(values[1].reshape(1,-1))
    pair2_FC = torch.Tensor(values[2]).reshape(1,-1)
    pair2_age = torch.Tensor(values[3].reshape(1,-1))
    nega_FC = torch.Tensor(values[4]).reshape(1,-1)
    nega_age = torch.Tensor(values[5].reshape(1,-1))
    
    return pair1_FC, pair1_age, pair2_FC, pair2_age, nega_FC, nega_age

def MakeInputBatch(Triplets):
    flag = 1
    pair1_FCs = Tensor()
    pair1_ages = Tensor()
    pair2_FCs = Tensor()
    pair2_ages= Tensor()
    nega_FCs= Tensor()
    nega_ages = Tensor()
    for i in range( len(Triplets) ):
        pair1_FC, pair1_age, pair2_FC, pair2_age, nega_FC, nega_age = DisTriplet(Triplets[i])
        if flag:
            pair1_FCs = pair1_FC
            pair1_ages = pair1_age
            pair2_FCs = pair2_FC
            pair2_ages = pair2_age
            nega_FCs = nega_FC
            nega_ages = nega_age
            flag = 0
        else:
            pair1_FCs = torch.cat([pair1_FCs, pair1_FC], dim = 0) #according to row
            #print(pair1_FCs.size())
            pair1_ages = torch.cat([pair1_ages, pair1_age], dim = 0)
            pair2_FCs = torch.cat([pair2_FCs, pair2_FC], dim = 0)
            pair2_ages = torch.cat([pair2_ages, pair2_age], dim = 0)
            nega_FCs = torch.cat([nega_FCs, nega_FC], dim = 0)
            nega_ages = torch.cat([nega_ages, nega_age], dim = 0)
           
    return pair1_FCs.to(device), pair1_ages.to(device), pair2_FCs.to(device),\
           pair2_ages.to(device), nega_FCs.to(device), nega_ages.to(device)

#--------------------for test-----------------------------------
if __name__ == '__main__':
  #multi_time_subs, multi_time_subs_pos = findMultiTimeSubinTeRealIDs('MNBCP229768')
  #print(multi_time_subs)
  #print(type(multi_time_subs_pos[0]))
  test_input = torch.randn(1, 180901).to(device)

  print( type(test_input) )

  FC = FCRecovery(test_input,mode = 'tensor')
  print('type (FC)',type(FC))
