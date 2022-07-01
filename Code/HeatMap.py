import numpy as np
from numpy.lib.type_check import real
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import math
import os
import pandas as pd
import copy

showTopK = int(68 * 68 * 0.1)
GAP = 100
FC_recon_path = './ExpResults/Exp1_unit_count_1500ReLU2_pccLoss100'  #202

files = os.listdir(FC_recon_path)
files_co = copy.deepcopy(files)
temp = []
temp_FC = []

ID_record = []
MultiTimeSubsPcc = 0
cnt_MultiTimeSubsPcc = 0
FC = np.zeros((68,68))
for filename in files:
  if(filename.split('.')[0] != 'pred_ori_corr' and filename.split('.')[-1] == 'txt'):

    realID_recon = filename.split('.')[0].split('_')[0]
    #day_recon = filename.split('.')[0].split('_')[1]
    #print('ID_recon is {0}, day_recon is {1}'.format(realID_recon,day_recon))
    #FC_recon = np.loadtxt(FC_recon_path + '/' + filename)
    
    if (realID_recon in ID_record):
      continue
    ID_record.append(realID_recon)

    for filename_co in files_co:
        realID_recon_co = filename_co.split('.')[0].split('_')[0]
        if (realID_recon_co == realID_recon):
          print(filename_co)
          day_recon_co = filename_co.split('.')[0].split('_')[1]
          print('ID_recon is {0}, day_recon is {1}'.format(realID_recon_co,day_recon_co))
          FC_recon = np.loadtxt(FC_recon_path + '/' + filename_co)

          
          #find the original file
          file_path = '/media/shawey/SSD8T/UNC_FC_Traverse/ROI70/ROI70_FC_AP'
          for root, dirs, files in os.walk(file_path):
              for file_name in files:
                  #print(file_name)
                  if(file_name.split('.')[1] == 'txt'):
                    realID = file_name.split('.')[0].split('_')[0]
                    days = file_name.split('.')[0].split('_')[1]
                    if(realID == realID_recon and int(day_recon_co) == int( np.floor( int(days) / GAP) )  ):          
                      print('real ID is {0}, days is {1}'.format(realID, days))
                      FC = np.loadtxt(file_path+'/'+file_name)
                      break
          
          if (FC.any()):
            if (temp):
              print('FC_recon prev and FC_recon dis is', np.mean(abs(temp[-1]-FC_recon)))
              print('FC_recon prev and FC_recon corr is', np.corrcoef(temp[-1].flatten(), FC_recon.flatten())[0,1])
            temp.append(FC_recon)
              
            FC_recon_fla = FC_recon.flatten()
            FC = abs(FC)
            FC_fla = FC.flatten()

            fisher_FC = (np.arctan(FC))
            fisher_FC_recon =( np.arctan(FC_recon) )
            FC_recon_thre = np.where(FC_recon>0.4, FC_recon, 0 )  #FC_recon
            FC_thre = np.where(FC>0.4, FC, 0 )  #FC
  
            #print('FC_recon min and max is {0} {1}, FC {2} {3}'.format(min(FC_recon_fla), max(FC_recon_fla), min(FC_fla), max(FC_fla)))

            if (temp_FC):
              print('FC prev and FC dis is', np.mean(abs(temp_FC[-1]-FC)))
              print('FC prev and FC corr is', np.corrcoef(temp_FC[-1].flatten(), FC.flatten())[0,1])
              MultiTimeSubsPcc +=  np.corrcoef(temp_FC[-1].flatten(), FC.flatten())[0,1]
              cnt_MultiTimeSubsPcc += 1
            temp_FC.append(FC)
              
            FC_FC_recon_dis = abs(FC-FC_recon).flatten()
            print('FC and FC_recon dis is', np.mean(abs(FC-FC_recon)))
            print('FC and FC_recon corr is', np.corrcoef(FC_fla, FC_recon_fla)[0,1])
            print('FC_thre and FC_recon_thre corr is', np.corrcoef(FC_recon_thre.flatten(), FC_thre.flatten())[0,1])
            fig = plt.figure()
            ax = fig.add_subplot(321)
            sns.heatmap( FC_recon_thre, fmt="d",   cmap = 'YlGnBu_r') #, cbar_kws = {'ticks':np.arange(-1.0,1.01,0.5)} vmin = 0.0,
            plt.xticks(())
            plt.yticks(())

            ax = fig.add_subplot(323)
            sns.heatmap(FC_thre, fmt="d",  cmap = 'YlGnBu_r') #vmin = 0.0,

            plt.xticks(())
            plt.yticks(())

            ax = fig.add_subplot(324)
            sns.heatmap(FC, fmt="d", cmap = 'YlGnBu_r' )  #YlGnBu_rcoolwarm

            plt.xticks(())
            plt.yticks(())

            ax = fig.add_subplot(322)
            sns.heatmap(FC_recon, fmt="d", cmap = 'YlGnBu_r' )
          
            plt.xticks(())
            plt.yticks(())

            ax = fig.add_subplot(325)
            plt.xlim(xmax=1,xmin=-0.5)
            plt.ylim(ymax=1,ymin=-0.5)
            a=np.arange(-1,2,0.01)
            plt.plot(a,a,'r')
            plt.plot(FC_recon_fla,FC_fla,'bo')
    
            ax = fig.add_subplot(326)
            plt.xlim(xmax=1,xmin=0)
            plt.hist(FC_recon_fla, bins = 20, alpha = 0.5)
            plt.hist(FC_fla, bins = 20, alpha = 0.5)
          
            #plt.xticks(())
            #plt.yticks(())
            FC = np.zeros((68,68))

            plt.xticks([])
            plt.yticks([])
            plt.show()

          else:

            fig = plt.figure()
            ax = fig.add_subplot(321)
            sns.heatmap( FC_recon, fmt="d",   cmap = 'YlGnBu_r') #, cbar_kws = {'ticks':np.arange(-1.0,1.01,0.5)} vmin = 0.0,
            if (temp):
              print('FC_recon prev and FC_recon corr is', np.corrcoef(temp[-1].flatten(), FC_recon.flatten())[0,1])
            temp.append(FC_recon)
            plt.xticks([])
            plt.yticks([])

            plt.show()

print('MultiTimeSubsPcc ', MultiTimeSubsPcc/cnt_MultiTimeSubsPcc)
#MultiTimeSubsPcc, cnt_MultiTimeSubsPcc
            

