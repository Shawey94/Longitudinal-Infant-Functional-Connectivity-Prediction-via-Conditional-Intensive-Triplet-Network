import numpy as np
import os
import pandas as pd
from Parse_args import *
from NNFunctions import *
import copy
import random

def StrConvertArray(lines):
    dataset = []
    for line in lines:
        line = line.strip('\n').split()
        #print(type(line[0]))
        temp = map(float,line)
        dataset.append(list(temp))
        #print((dataset))
    return dataset


#ATTENTION! filter the subjects timescans that exceed the label.      
def ReadFCData_MulTimeSubs_ABS(file_path):
    te_FCs = []
    te_RealIDs = []
    te_Days = []
    tr_FCs = []
    tr_RealIDs = []
    tr_Days = []
    f = open(opt.MultiTimeSubFile,"r")    #subs that have multiple times scans
    lines = f.readlines()
    f.close()
    lines = StrProcess(lines)
    teSub_cnt = opt.TestSubs
    subs = []
    files = os.listdir(file_path)
    random.shuffle(files)
    random.shuffle(files)
    #print('type files' ,type(files))
    co_files = copy.deepcopy(files)
    days_record = []
    arr_ind = np.triu_indices(opt.ROI_num, k = 1)
    for file_name in files:
        co_files.remove(file_name)
        #print(file_name)
        realID = file_name.split('.')[0].split('_')[0]
        days = int(file_name.split('.')[0].split('_')[1])   
        if(days > 600):  # [0,600) days
           continue
        #print(days)
        if(np.floor(days) not in days_record):
           days_record.append(np.floor(days))
        #print(sampleID)
        FC_temp = np.loadtxt(file_path+'/'+file_name)
        # take the UpTriangle and flatten to 1-D
        UpTriangle = abs(FC_temp[arr_ind])
        #UpTriangle = (FC_temp[arr_ind])
        #UpTriangle = np.triu(FC_temp,1).flatten()
        #UpTriangle = UpTriangle[UpTriangle != 0]  
        #UpTriangle = abs(UpTriangle)
        if (realID in lines):
            if (len(subs) < teSub_cnt and realID not in subs):
                subs.append(realID)
                te_FCs.append(UpTriangle)
                te_RealIDs.append(realID)
                te_Days.append(days)
                for co_file in co_files:
                    co_realID = co_file.split('.')[0].split('_')[0]
                    if(co_realID == realID):
                        days = int(co_file.split('.')[0].split('_')[1])
                        if(days > 600):  # [0,600) days
                            continue
                        #print(sampleID)
                        FC_temp = np.loadtxt(file_path+'/'+co_file)
                        # take the UpTriangle and flatten to 1-D
                        UpTriangle = abs(FC_temp[arr_ind])
                        #UpTriangle = (FC_temp[arr_ind])
                        te_FCs.append(UpTriangle)
                        te_RealIDs.append(realID)
                        te_Days.append(days)
            elif(realID not in subs):
                tr_FCs.append(UpTriangle)
                tr_RealIDs.append(realID)
                tr_Days.append(days)
        else:
            tr_FCs.append(UpTriangle)
            tr_RealIDs.append(realID)
            tr_Days.append(days)
        #print(FCInfo)
    print('days_record len is ', len(days_record))
    return tr_FCs, tr_RealIDs, tr_Days, te_FCs, te_RealIDs, te_Days



#--------------------for test-----------------------------------
if __name__ == '__main__':
  file_path = '/media/shawey/SSD8T/UNC_FC_Traverse/ROIs360/ROI360_FC_IDAgeGroup_AP'
  tr_FCs, tr_RealIDs, tr_Days, te_FCs, te_RealIDs, te_Days = ReadFCData_MulTimeSubs_ABS(file_path)
  print('tr_FC len is',len(tr_FCs))
  print('te_FC len is',len(te_FCs))
  print('te_realIDs are', te_RealIDs)
  print('te_days are', te_Days)
  print('tr_realIDs are', tr_RealIDs)
  print('tr_days are', tr_Days)
