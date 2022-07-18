#Reconstruc Self
#Reconstrct with Different IDs within the same subject
import torch 
import numpy as np
import time
from torch.autograd import Variable
from CustomedDataLoder import *
from Parse_args import *
from NNClasses_resi import *
from NNFunctions import *
import torch.utils.data as Data
import itertools
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau


GAP = 100
loss_coeff = [10, 0.05, 0, 100, 5]   #0.1*triplet_loss + 0.1*age_loss + 0.1 * noise_loss + recon_loss + cross_recon_loss + pcc_loss

if __name__ == '__main__':

    StartTime = time.time()
    torch.manual_seed(0)

    valid = Variable(FloatTensor(opt.batch_size, 1).fill_(1.0), requires_grad=False)
    fake = Variable(FloatTensor(opt.batch_size, 1).fill_(0.0), requires_grad=False)

    save_index = 4  #add more layer in residualblock
    for RT in range(opt.RepTime):   #Repteat Experiments
        count = 500
        tr_FCs, tr_RealIDs, tr_Days, te_FCs, te_RealIDs, te_Days = ReadFCData_MulTimeSubs_ABS(opt.FC_path)
        tr_FCs = np.array(tr_FCs)
        tr_Days = np.array( list((map(float, tr_Days))) )
       
        tr_Days = np.floor( tr_Days /GAP)     #convert to age group, ex 107 days is group 1
        print(tr_Days)
        print('tr_FCs len is {0}, tr_RealIDs len {1}, tr_Days len {2}'.format(len(tr_FCs), len(tr_RealIDs), len(tr_Days)))
        #with open('set_tr_realIDs.txt','a+') as f:    #设置文件对象
        #     f.write(str(set(tr_RealIDs)))                 #将字符串写入文件中
        #     f.write('\n')

        te_FCs = np.array(te_FCs)
        te_Days = np.array( list((map(float, te_Days))) )
       
        te_Days = np.floor( te_Days /GAP)     #convert to age group, ex 107 days is group 1
        print(te_Days)
        print('te_FCs len is {0}, te_RealIDs len {1}, te_Days len {2}'.format(len(te_FCs), len(te_RealIDs), len(te_Days)))
        #print('te_FCs one element is ', te_FCs[0]) 

        DataLoaderTime = time.time()
        print('DataLoader time is ', DataLoaderTime-StartTime)

        save_index += 1
        exp_dir = './ExpResults/Exp' + str(save_index)+ '_unit_count_'+str(opt.unit_count)+'ReLU2_pccLoss'+ str(loss_coeff[3])
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        #MLRegression(tr_FCs, tr_RealIDs, tr_Days, te_FCs, te_RealIDs, te_Days, exp_dir)
        
        TripletList, NumTriplet = GeneTriplet(tr_FCs, tr_RealIDs, tr_Days)
        #print(sys.getsizeof(TripletList))
        print('NumTriplet is', NumTriplet)
        fir_ele_key = int(list(TripletList[0].keys())[0])
        embedding_net = EmbeddingNet(int((TripletList[0][fir_ele_key][0]).shape[0]))
        reconNet = ReconNet(int((TripletList[0][fir_ele_key][0]).shape[0]))
        dis_noise = Discriminator_noise()
        siameseNet_triplet = SiameseNet_triplet(embedding_net)
        age_predictor = AgePredictor()
        #dis_FC = Discriminator_FC()
 
        '''
        embedding_net = nn.DataParallel(embedding_net)
        reconNet = nn.DataParallel(reconNet)
        dis_noise = nn.DataParallel(dis_noise)
        age_predictor = nn.DataParallel(age_predictor)
        '''

        embedding_net.to(device)
        reconNet.to(device)
        dis_noise.to(device)
        age_predictor.to(device)
        #dis_FC.to(device)
 

        embedding_net.apply(weights_init_normal)          
        reconNet.apply(weights_init_normal)           
        dis_noise.apply(weights_init_normal)            
        age_predictor.apply(weights_init_normal)

        distant_loss = torch.nn.L1Loss()
        adversarial_loss = torch.nn.BCEWithLogitsLoss()

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,itertools.chain(embedding_net.parameters(), reconNet.parameters(), age_predictor.parameters())),
                                        lr=opt.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay =0)
            
        optimizer_D = torch.optim.Adamax(filter(lambda p: p.requires_grad, itertools.chain(dis_noise.parameters())),
                                        lr=opt.lr, betas=(0.9, 0.999), eps=1e-08)
        #optimizer_dis_FC = torch.optim.Adamax(filter(lambda p: p.requires_grad, itertools.chain(dis_FC.parameters())), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08)
        scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=20, verbose=True)
        scheduler_D = ReduceLROnPlateau(optimizer_D, 'min',factor=0.5, patience=20, verbose=True)
        #scheduler_dis_FC = ReduceLROnPlateau(optimizer_dis_FC, 'min',factor=0.5, patience=20, verbose=True)
        
        constr_loss = np.zeros((3, opt.n_epochs * int(np.ceil(NumTriplet/opt.batch_size))))   #Subject1 T1, Subject1 T2, subject2 T, subject1(recon) T1, subject1(recon) T2
        te_losses = []
        constr_index = 0

        PCC_loss = np.zeros((3, opt.n_epochs * int(np.ceil(NumTriplet/opt.batch_size))))  

        optimizer.zero_grad()
        optimizer_D.zero_grad()
        #optimizer_dis_FC.zero_grad()
        embedding_net.train()
        reconNet.train()
        dis_noise.train()
        age_predictor.train()
        #dis_FC.train()

    
        epoch_loss = []
        for epoch in range(opt.n_epochs): #for each fold in K-fold, repeat n_epochs times
            print('int(np.ceil(NumTriplet/opt.batch_size))', int(np.ceil(NumTriplet/opt.batch_size)))
            temp_overall_loss = 0
            for NT in range( int(np.ceil(NumTriplet/opt.batch_size)) ):
                print('epoch {0}, NT is {1}'.format(epoch,NT))
                if (NT+1)*opt.batch_size > NumTriplet:                
                   break
                else:
                    tr_Triplets = TripletList[NT*opt.batch_size:(NT+1)*opt.batch_size]
                #if(count < 499):
                #    break
                #count -= 1
                pair1_FCs, pair1_ages, pair2_FCs, pair2_ages, nega_FCs, nega_ages = MakeInputBatch(tr_Triplets)
                #print(pair1_FCs.size())

                valid = Variable(FloatTensor(len(pair1_ages), 1).fill_(1.0), requires_grad=False)
                fake = Variable(FloatTensor(len(pair1_ages), 1).fill_(0.0), requires_grad=False)

                emb1, emb2, emb_nega, emb1_age, emb2_age, emb_nega_age, multi_layer_feature1, multi_layer_feature2, multi_layer_feature_x = siameseNet_triplet(pair1_FCs, pair2_FCs, nega_FCs)
                relation_pair1_pair2, relation_pari1_nega, relation_pair2_nega = Triplet_corr(emb1, emb2, emb_nega)

                print('same sub corr {0}, diff1 {1}, diff2 {2}'.format(corr_loss(emb1, emb2).detach(), corr_loss(emb1, emb_nega).detach(),  corr_loss(emb2, emb_nega).detach() ))

                recon_pair1 = reconNet(emb1, pair1_ages)
                #FC_recon_pair1 = FCRecovery(recon_pair1,mode= 'tensor')
                recon_pair2 = reconNet(emb2, pair2_ages)
                #FC_recon_pair2 = FCRecovery(recon_pair2,mode= 'tensor')
                recon_nega = reconNet(emb_nega, nega_ages)
                #FC_recon_nega = FCRecovery(recon_nega,mode= 'tensor')

                Age_esti1 = age_predictor(emb1_age)
                Age_esti2 = age_predictor(emb2_age)
                Age_esti_nega = age_predictor(emb_nega_age)

                triplet_loss = TripleLoss(relation_pair1_pair2, relation_pari1_nega, relation_pair2_nega) / len(emb1)
                age_loss = AgeLoss(Age_esti1, pair1_ages) + AgeLoss(Age_esti2, pair2_ages) + AgeLoss(Age_esti_nega, nega_ages) 

                #noise_loss = (adversarial_loss( dis_noise(noise1), valid)  + adversarial_loss( dis_noise(noise2), valid ) + adversarial_loss( dis_noise(noise_nega), valid ) )/3
                recon_loss = 1/3*((distant_loss(recon_pair1.to(device), pair1_FCs.to(device)) + distant_loss(recon_pair2.to(device), pair2_FCs.to(device)) + distant_loss(recon_nega.to(device), nega_FCs.to(device)) ))
            
                #input('FC DIS LOSS press any key') CPU usage 6.5 Gb
                #print(recon_pair1.device, pair1_ages.device) #both cuda:0
                if torch.max(torch.isnan(recon_pair1)):
                    print('NaN occurs in TripletLoss, emb init is 0')
                    break

                #FC_dis_loss =   1/3 * adversarial_loss( dis_FC(recon_pair1, pair1_ages),valid) 
                #del inputs_adju
                #inputs_adju = FCRecovery(recon_pair2,mode= 'tensor').to(device)  #use PCA to compress img
                #FC_dis_loss += 1/3 * adversarial_loss( dis_FC(recon_pair2, pair2_ages),valid) 
                #del inputs_adju
                #inputs_adju = FCRecovery(recon_nega,mode= 'tensor').to(device)  #use PCA to compress img
                #FC_dis_loss += 1/3 * adversarial_loss( dis_FC(recon_nega, nega_ages),valid)  
                pcc_loss = 1/3*(1- corr_loss(recon_nega, nega_FCs)+  1- corr_loss(recon_pair1, pair1_FCs)+1- corr_loss(recon_pair2, pair2_FCs) )
                 
                
                overall_loss = loss_coeff[0]*triplet_loss.to(device) + loss_coeff[1]*age_loss.to(device) + loss_coeff[3] * recon_loss +  loss_coeff[4] * pcc_loss 
                #loss_coeff[2] * noise_loss.to(device)
                temp_overall_loss += overall_loss.detach()
                #optimizer.step()

                #optimizer_D.zero_grad()
                #z = Variable(Tensor(np.random.normal(0, 1, (len(pair1_ages), opt.noiseSize))))
                #d_real_loss = adversarial_loss(dis_noise(z), valid)
                #d_fake_loss = (adversarial_loss(dis_noise(noise1), fake) + adversarial_loss(dis_noise(noise2), fake) + adversarial_loss(dis_noise(noise_nega), fake))/3
                #d_loss = (d_real_loss + d_fake_loss) / 2
                
                #CVPR 2021, Least-square GANs
                #Pre_Ori_real_loss =  (adversarial_loss( dis_FC(pair1_FCs, pair1_ages),valid)  + adversarial_loss( dis_FC(pair2_FCs, pair1_ages),valid)  + adversarial_loss( dis_FC(nega_FCs, pair1_ages),valid ) )/3
                #Pre_Ori_fake_loss = ( adversarial_loss( dis_FC(recon_pair1, pair1_ages),fake ) + adversarial_loss( dis_FC(recon_pair2, pair2_ages),fake) + adversarial_loss( dis_FC(recon_nega, nega_ages),fake) )/3
                #Pre_Ori_loss = (Pre_Ori_real_loss + Pre_Ori_fake_loss) / 2


                overall_loss.backward()              #retain_graph=True
                scheduler.step(overall_loss)                
                #Pre_Ori_loss.backward(retain_graph = True)
                #scheduler_dis_FC.step(Pre_Ori_loss)
                #d_loss.backward()
                #scheduler_D.step(d_loss)  
                optimizer.step()  
                #optimizer_dis_FC.step()    
                #optimizer_D.step()
                
                print('Triplet loss is ',triplet_loss.detach())
                print('recon loss {0}, {1}, {2}'.format( (distant_loss(recon_pair1, pair1_FCs)).detach(), (distant_loss(recon_pair2, pair2_FCs)).detach(),  (distant_loss(recon_nega, nega_FCs)).detach()  ))
               
                print('age loss {0}, {1}, {2}'.format( (AgeLoss(Age_esti1, pair1_ages)).detach(), (AgeLoss(Age_esti2, pair2_ages)).detach(), (AgeLoss(Age_esti_nega, nega_ages)).detach() ))
                #print('real and fake noise loss {0}, {1}'.format( d_real_loss.detach(), d_fake_loss.detach()) )
                #print('FC_dis_loss {0}, real and fake FC loss {1}, {2}'.format(FC_dis_loss.detach(), Pre_Ori_real_loss.detach(), Pre_Ori_fake_loss.detach()) )
                temp_nega = FCRecovery(recon_nega.detach(), 'tensor')
                temp_1 = FCRecovery(recon_pair1.detach(), 'tensor')
                temp_2 = FCRecovery(recon_pair2.detach(), 'tensor')
                corr1 = corr_loss(temp_nega.flatten(), FCRecovery(nega_FCs, 'tensor').flatten() ).detach()
                corr2 = corr_loss(temp_1.flatten(), FCRecovery(pair1_FCs, 'tensor').flatten()).detach()
                corr3 = corr_loss(temp_2.flatten(), FCRecovery(pair2_FCs, 'tensor').flatten()).detach()
                print('With CorrLoss Corr {0}, {1}, {2}'.format( corr1,  corr2, corr3 ))

                PCC_loss[0,constr_index] += corr1
                PCC_loss[1,constr_index] += corr2
                PCC_loss[2,constr_index] += corr3

                '''
                print('With CorrLoss Corr {0}, {1}, {2}'.format( corr_loss(recon_nega, nega_FCs).detach().cpu(),  corr_loss(recon_pair1, pair1_FCs).detach(), corr_loss(recon_pair2, pair2_FCs).detach() ))

                PCC_loss[0,constr_index] +=(corr_loss(recon_nega, nega_FCs).detach()) 
                PCC_loss[1,constr_index] +=(corr_loss(recon_pair1, pair1_FCs).detach())
                PCC_loss[2,constr_index] +=(corr_loss(recon_pair2, pair2_FCs).detach())
                '''

                constr_loss[0,constr_index] +=((distant_loss(recon_pair1, pair1_FCs)).detach())
                constr_loss[1,constr_index] +=((distant_loss(recon_pair2, pair2_FCs)).detach())
                constr_loss[2,constr_index] +=((distant_loss(recon_nega, nega_FCs)).detach())
              
                constr_index += 1
                torch.cuda.empty_cache()
            epoch_loss.append(temp_overall_loss / int(np.ceil(NumTriplet/opt.batch_size)) )
        plt.plot(epoch_loss, label = 'overall loss')
        plt.savefig(exp_dir+'/'+str(save_index)+'_'+str(epoch)+'_overall_loss_tr.png')
        plt.close('all')


        plt.plot(PCC_loss[0,:-1],label = 'T1 pcc')
        plt.plot(PCC_loss[1,:-1],label = 'T2 pcc')
        plt.plot(PCC_loss[2,:-1],label = 'Nega pcc')
        plt.legend()
        plt.xlabel('iterations')
        plt.ylabel('pcc') 
        plt.savefig(exp_dir+'/'+str(save_index)+'_'+str(epoch)+'_pcc_tr.png')
        plt.close('all')

        plt.plot(constr_loss[0,:-1],label = 'Subject1 T1 constrction')
        plt.plot(constr_loss[1,:-1],label = 'Subject1 T2 constrction')
        plt.plot(constr_loss[2,:-1],label = 'Subject2 T constrction')
      
        plt.legend()
        plt.xlabel('iterations')
        plt.ylabel('MAE')
        
        #y_major_locator=MultipleLocator(0.20)
        #ax=plt.gca()
        #ax.yaxis.set_major_locator(y_major_locator)
        plt.savefig(exp_dir+'/'+str(save_index)+'_'+str(epoch)+'_tr.png')
        plt.close('all')
        #plt.show()


        #TESTING
        with torch.no_grad():
            embedding_net.eval()
            reconNet.eval()
            dis_noise.eval()
            age_predictor.eval()  
            tested_IDs = {}
            te_RealIDs_copy = copy.deepcopy(te_RealIDs)
            for tr_ite in range(len(te_FCs)):
                te_one_FC = te_FCs[tr_ite]
                te_one_Day = te_Days[tr_ite]
                te_RealID = te_RealIDs[tr_ite]
                if (te_RealID not in tested_IDs):
                    tested_IDs[te_RealID] = []
                if (te_one_Day not in tested_IDs[te_RealID]):
                    tested_IDs[te_RealID].append(te_one_Day)     
                    te_ID_pccs = []

                    te_Day = ((torch.as_tensor(te_one_Day)).reshape(1,-1)).to(device)
                    te_FC = ((torch.as_tensor(te_one_FC)).reshape(1,-1)).to(device)
                    te_feature_age, te_feature_ID, te_multi_layer_feature = embedding_net(te_FC.float())
                    recon_test = reconNet(te_feature_ID, te_Day)     
        
                    #if (tr_ite in multi_time_subs_pos):
                    FC_recon = FCRecovery(recon_test.detach().cpu().numpy())

                    np.savetxt(exp_dir+'/'+str(te_RealID)+'_'+str(te_one_Day)+'_recon_inputFC.txt',FC_recon)
                    corr_pred_ori = (np.corrcoef(recon_test.detach().cpu().numpy(), te_FC.cpu().numpy())[0,1])
                    file_r_pred_ori=open(exp_dir+'/'+'pred_ori_corr.txt',mode='a+')
                    file_r_pred_ori.write(str(corr_pred_ori) + '\n')
                    file_r_pred_ori.close()
                    te_loss = distant_loss(recon_test, te_FC)
                    te_losses.append(te_loss.detach().cpu().numpy())
                    
                    #for co_ite in range(1, len(te_FCs), 1):
                        #te_RealID_co = te_RealIDs[co_ite]
                        #te_one_Day_co = te_Days[co_ite]
                        #te_Day_co = ((torch.as_tensor(te_one_Day_co)).reshape(1,-1)).to(device)
                    for i in range( len(opt.age_group)):
                        te_Day_co = ((torch.as_tensor(i)).reshape(1,-1)).to(device)
                        if(te_Day_co != te_Day):
                            #te_one_FC_co = te_FCs[co_ite]
                            #te_FC_co = ((torch.as_tensor(te_one_FC_co)).reshape(1,-1)).to(device)
                            #te_feature_age_co, te_feature_ID_co, te_multi_layer_feature_co = embedding_net(te_FC_co.float())
                            #te_pcc = (np.corrcoef(te_feature_ID_co.detach().cpu().numpy(), te_feature_ID.cpu().numpy())[0,1])
                            #te_ID_pccs.append(te_pcc)

                            recon_test_co = reconNet(te_feature_ID, te_Day_co)   
                            #min_value = torch.min(recon_test_co) 
                                            

                            #if (tr_ite in multi_time_subs_pos):
                            FC_recon_co = FCRecovery(recon_test_co.detach().cpu().numpy())
                            #np.savetxt(opt.FCreconSavePath+'/'+str(te_RealID)+'_'+str(te_one_Day)+'_reconseq.txt',recon_test.detach().cpu().numpy())
                            np.savetxt(exp_dir+'/'+str(te_RealID)+'_'+str(i)+'_BaseOn_'+str(te_one_Day)+'_recon.txt',FC_recon_co)
                            #corr_pred_ori = (np.corrcoef(recon_test_co.detach().cpu().numpy(), te_FC_co.cpu().numpy())[0,1])
                            #file_r_pred_ori=open(exp_dir+'/'+'pred_ori_corr.txt',mode='a+')
                            #file_r_pred_ori.write(str(corr_pred_ori) + '\n')
                            #file_r_pred_ori.close()
                            #te_loss = distant_loss(recon_test_co, te_FC_co)
                            #te_losses.append(te_loss.detach().cpu().numpy())


                    with open(exp_dir + '/' + 'te_ID_pccs.txt','a+') as f:    #设置文件对象
                        f.write(str(te_ID_pccs) )                 #将字符串写入文件中
                        f.write('\n')

        
        te_plt_x = [xx for xx in range(len(te_losses))]
        plt.scatter(te_plt_x, te_losses)
        #plt.legend()
        plt.ylabel('MAE')
        plt.xlabel('subjects')
        #y_major_locator=MultipleLocator(0.15)
        #ax=plt.gca()
        #ax.yaxis.set_major_locator(y_major_locator)
        plt.savefig(exp_dir+'/'+str(save_index)+'_'+str(epoch)+'_te.png')
        plt.close('all')
        #plt.show()

        del embedding_net
        del reconNet
        del dis_noise
        del age_predictor
        torch.cuda.empty_cache()
        KFoldTime = time.time()
        print('time is ', KFoldTime-StartTime)
        #input('input any key to continue')



