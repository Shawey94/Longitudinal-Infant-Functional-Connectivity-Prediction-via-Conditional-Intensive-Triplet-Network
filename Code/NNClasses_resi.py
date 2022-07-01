from typing import Tuple
import torch.nn as nn
from Parse_args import *
from NNFunctions import *
from sklearn.decomposition import PCA


class AgePredictor(nn.Module):
    def __init__(self):
        super(AgePredictor, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.ageSize, 32),   #tried 16(3.2 Error)
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            #nn.Linear(32, 8),
            #nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Linear(32,1)

    def forward(self, z):
        
        out = self.model(z)
        out2 = torch.nn.functional.softmax(out, dim = 1)  #every row possibility sums to 1
        age = self.layer1(out2)
        return age  

class SiameseNet_triplet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet_triplet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x):
        output1_age, output1, multi_layer_feature1= self.embedding_net(x1)
        output2_age, output2, multi_layer_feature2 = self.embedding_net(x2)
        output_x_age, output_x, multi_layer_feature_x = self.embedding_net(x)
        
        return output1, output2,  output_x, output1_age, output2_age,  output_x_age, multi_layer_feature1, multi_layer_feature2, multi_layer_feature_x

class EmbeddingNet(nn.Module):    
    def __init__(self,input_size):
        super(EmbeddingNet, self).__init__()   
    
        self.layer1 = nn.Sequential(nn.Linear(input_size, opt.latent_dim), nn.BatchNorm1d(opt.latent_dim), nn.LeakyReLU()) #for debug nn.PReLU(),
        self.layer2 = nn.Sequential(nn.Linear(opt.latent_dim, opt.latent_dim-opt.ageSize-opt.noiseSize), nn.BatchNorm1d(opt.latent_dim-opt.ageSize-opt.noiseSize), nn.LeakyReLU()) #for debug nn.PReLU(),
        self.layer3 = nn.Sequential(nn.Linear(opt.latent_dim, opt.ageSize), nn.BatchNorm1d(opt.ageSize), nn.LeakyReLU()) #for debug nn.PReLU(),
        #self.layer4 = nn.Sequential(nn.Linear(opt.latent_dim, opt.noiseSize), nn.BatchNorm1d(opt.noiseSize), nn.LeakyReLU()) #for debug nn.PReLU(),
        #self.mu = nn.Linear(opt.noiseSize, opt.noiseSize)
        #self.logvar = nn.Linear(opt.noiseSize, opt.noiseSize)
        
    def forward(self, x):
        latent_fea = self.layer1(x)
        #noise_temp = self.layer4(latent_fea)
        #mu = self.mu(noise_temp)
        #logvar = self.logvar(noise_temp)
        #noise = reparameterization(mu, logvar)
        
        feature_ID = self.layer2(latent_fea)
        feature_age = self.layer3(latent_fea)

        return feature_age, feature_ID, latent_fea  #noise, 

class ReconNet(nn.Module): 
    def __init__(self,input_size):
        super(ReconNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(opt.latent_dim,input_size), nn.BatchNorm1d(input_size), nn.ReLU())  #nn.Linear(opt.latent_dim,opt.latent_dim)  nn.LeakyReLU(0.3)
        #self.layer2 = nn.Sequential(nn.Linear(opt.latent_dim - opt.noiseSize - opt.ageSize,opt.latent_dim), nn.BatchNorm1d(opt.latent_dim), nn.ReLU(inplace=True))
        #self.layer3 = nn.Sequential(nn.Linear(8192,input_size), nn.BatchNorm1d(input_size), nn.ReLU(inplace=True),nn.Dropout(0.05))
        self.agingModule = AgingModule(age_group= len(opt.age_group), Dim_AgeConID = opt.latent_dim)  #self with weight init_normal
    def forward(self, emb, ages):
        ageConditonID = self.agingModule(emb, condition = ages)   # conditon >= 1 
        #x = torch.cat([ageConditonID.to(device), noise.to(device)], dim=1) + multi_layer_feature
        x = ageConditonID.to(device) 
        #x = self.layer2(emb)
        x1 = self.layer1(x)           
        return x1 


class Discriminator_noise(nn.Module):
    def __init__(self):
        super(Discriminator_noise, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.noiseSize, 30),
            nn.BatchNorm1d(30),
            nn.ReLU(inplace=True),            
            nn.Linear(30, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        validity = self.model(z)
        return validity 


class ResidualBlock(nn.Module):
    def __init__(self, unit_count, age_group, sigma):
        super(ResidualBlock, self).__init__()
        conv_dim = int((age_group - (age_group - 1) * sigma) * unit_count)
        self.conv1 = nn.Sequential(nn.Linear(unit_count, conv_dim), nn.BatchNorm1d(conv_dim))
        self.router1 = TaskRouter(unit_count, age_group, sigma)
        self.conv2 = nn.Sequential(nn.Linear(unit_count, conv_dim), nn.BatchNorm1d(conv_dim))
        #self.router2 = TaskRouter(unit_count, age_group, sigma)
        self.relu1 = nn.LeakyReLU()
        self.relu2 = nn.LeakyReLU()


    def forward(self, inputs):
        x, task_ids = inputs[0], inputs[1]
        x2 = self.router1( self.relu1(self.conv1(x)), task_ids)
        #print('1st x2 ',x2)
        x2 = self.router1( self.relu2(self.conv2(x2)), task_ids)
        return {0: x2, 1: task_ids}  #{0: x2 + x1, 1: task_ids}


class TaskRouter(nn.Module):

    def __init__(self, unit_count, age_group, sigma):
        super(TaskRouter, self).__init__()

        conv_dim = int((age_group - (age_group - 1) * sigma) * unit_count)
        #print('taskrouter conv_dim is', conv_dim)

        self.register_buffer('_unit_mapping', 0.0*torch.ones(age_group, conv_dim))
        # self._unit_mapping = torch.zeros((age_group, conv_dim))
        start = 0
        for i in range(age_group):
            self._unit_mapping[i, start: start + unit_count] = 1
            start = int(start + (1 - sigma) * unit_count)
        #print('unit_mapping is ',self._unit_mapping)
            #print('start is ',start)

    def forward(self, inputs, task_ids):
        new_inputs = torch.zeros( int(list(inputs.size())[0]), opt.unit_count).to(device)
        for i in range(len(task_ids)):
            #mask = torch.index_select(self._unit_mapping, 0, task_ids[i].long()) \
            #    .unsqueeze(2)
            #inputs[i,:,:] = inputs[i,:,:] * mask
            #print(int( task_ids[i]* int((1 - opt.sigma) * opt.unit_count) ) )
            #print('new_inputs[i,:] shape {0}, inputs[i,:] shape {1} '.format (new_inputs[i,:].shape, inputs[i,:].shape ) )
            #print('i {0}, task_ids[i] {1}'.format( i, task_ids[i] ))
            #print('index1 {0}, index2 {1}'.format( int( task_ids[i]* int((1 - opt.sigma) * opt.unit_count) ), int( task_ids[i]* int((1 - opt.sigma) * opt.unit_count) + opt.unit_count)))
            new_inputs[i,:] = inputs[i,:][int( task_ids[i]* int((1 - opt.sigma) * opt.unit_count) ): int( task_ids[i]* int((1 - opt.sigma) * opt.unit_count) + opt.unit_count) ]
        return new_inputs

class AgingModule(nn.Module):
    def __init__(self, age_group, Dim_AgeConID, repeat_num=6 ):
        super(AgingModule, self).__init__()
        layers = []
        sigma = opt.sigma
        unit_count = opt.unit_count
        conv_dim = int((age_group - (age_group - 1) * sigma) * unit_count)
        self.router = TaskRouter(unit_count, age_group, sigma)
        #print('aging conv_dim is ', conv_dim)
        self.conv1 = nn.Sequential(
            nn.Linear( opt.latent_dim - opt.noiseSize - opt.ageSize, conv_dim),
            nn.BatchNorm1d(conv_dim),
            nn.LeakyReLU(),
        )
        for _ in range(repeat_num):
            layers.append(ResidualBlock(unit_count, age_group, sigma))
        self.transform = nn.Sequential(*layers)
        self.conv2 = nn.Sequential(
            nn.Linear(unit_count, Dim_AgeConID),
            nn.BatchNorm1d(Dim_AgeConID),
            nn.LeakyReLU(),
        )

        self.__init_weights()

    def forward(self, x_id, condition):
        #print(x_id)
        x_id = self.conv1(x_id)
        x_id = self.router(x_id, condition)
        #print('after router',x_id)
        inputs = {0: x_id, 1: condition}
        x = self.transform(inputs)[0]
        x = self.conv2(x)
        return x 

    def __init_weights(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.fill_(0)
