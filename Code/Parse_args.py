# 2200 100 100
import argparse
import torch

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = torch.device('cuda' if cuda else 'cpu')

if torch.cuda.device_count() > 1:
          print("Let's use", torch.cuda.device_count(), "GPUs!")

parser = argparse.ArgumentParser()
parser.add_argument('--RepTime', type=int, default=10, help='number of repeat times')
parser.add_argument('--n_epochs', type=int, default=6, help='number of epochs of training')
#parser.add_argument('--thre', type=int, default=100, help='number of negative subjects chosen to generate triplets')
parser.add_argument('--batch_size', type=int, default=100, help='size of the batches')  # not enough CPU or GPU for big batch size
parser.add_argument('--lr', type=float, default=0.001, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--latent_dim', type=int, default=4000, help='dimensionality of the latent space')
parser.add_argument('--noiseSize', type=int, default=200, help='latent noise dimension')
parser.add_argument('--ageSize', type=int, default=500, help='latent age dimension')
parser.add_argument('--FC_path', type=str, default='/media/shawey/SSD8T/UNC_FC_Traverse/ROI70/ROI70_FC_AP', help= 'FC stored path')
parser.add_argument('--age_group',type=list, default=[0,1,2,3,4,5],help='group of age, 0 means 0-29 days, 1 means 30-59 days, etc..')
parser.add_argument('--sigma', type=float, default=0.0, help='aging module coincidence rate') # 0.1 0.5
parser.add_argument('--unit_count', type=int, default=1500, help='aging module')  #16,24, 
#parser.add_argument('--pics_save',type=str, default='/media/shawey/SSD8T/UNC_FC_Traverse/FCTraverseCode/GPU_Version_WithICMV3_UseMultiGPUs_AgeMerging_FCDiscrimintor_30Days/pics_save',help='path for saving pictures')
#parser.add_argument('--FCreconSavePath', type=str,default='/media/shawey/SSD8T/UNC_FC_Traverse/FCTraverseCode/GPU_Version_WithICMV3_UseMultiGPUs_AgeMerging_FCDiscrimintor_30Days/FCRecon', help='dir to save reconstrcted FCs')
parser.add_argument('--MultiTimeSubFile', type=str, default='/media/shawey/SSD8T/UNC_FC_Traverse/ROI70/mul_time_subjects_AP.txt', help='The file that stores subjects with multiple timepoints')

parser.add_argument('--ROI_num', type=int, default=68, help='ROI Number')

parser.add_argument('--TestSubs', type=int, default=10, help='Testing Subjects Number')

opt = parser.parse_args()
