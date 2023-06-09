'''
Created by Victor Delvigne
ISIA Lab, Faculty of Engineering University of Mons, Mons (Belgium)
victor.delvigne@umons.ac.be

Source: Bashivan, et al."Learning Representations from EEG with Deep Recurrent-Convolutional Neural Networks." International conference on learning representations (2016).

Copyright (C) 2019 - UMons

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
'''
from ast import Tuple
import pickle
import typing
import numpy as np
import scipy.io as sio
import torch
import os
from os import path

import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader,random_split
from myUtils.utils.Utils import *
from myUtils.utils.Models import *
import myUtils.utils.Models as mods

torch.manual_seed(1234)
np.random.seed(1234)

import warnings
warnings.simplefilter("ignore")
# from prefect import flow, task

# @task(name="generate_images")
def generate_images(n_freq, n_epoch,directory:str = None,feats=None,locs=None):
    if(directory is not None):
        if(os.path.exists(directory)):
            return sio.loadmat(directory)["img"]
    return create_img(feats,locs,n_freq, n_epoch,32,directory) #return Images array
        
# @task
def loadData(Features: list = None, Patient_id: list = None, Location:list = None, columns:str = None) :
    """
    Description : 
        Load all the data using the path to the .mat files
    Parameters:
        Features: str
        Patient_id: str
        Location:str
    Returns:
        np.arrays
        
    """
    # if(Images is None):
    #     Images = sio.loadmat("Sample Data/images_time.mat")["img"] #corresponding to the images mean for all the seven windows
    # else:
    #     Images = sio.loadmat(Images)['img']    
    # Mean_Images = np.mean(Images, axis= 0)

    if(Features is None): 
        #Mean_Images = sio.loadmat("Sample Data/images.mat")["img"] #corresponding to the images mean for all the seven windows
        Label = (sio.loadmat("Sample Data/FeatureMat_timeWin")["features"][:,-1]-1).astype(int) #corresponding to the signal label (i.e. load levels).
        Feats = sio.loadmat('Sample Data/FeatureMat_timeWin.mat')['features']
    else:
        Label = (Features[:,-1]).astype(int) #(sio.loadmat(Features)['features']
        Feats = Features#sio.loadmat(Features)['features']

    if(Patient_id is None):
        Patient_id = sio.loadmat("Sample Data/trials_subNums.mat")['subjectNum'][0] #corresponding to the patient id
    else:
        Patient_id = Patient_id#sio.loadmat(Patient_id)["subjectNum"][0]
        
    if(Location is None):
        locs3D = sio.loadmat('Sample Data/Neuroscan_locs_orig.mat')['A']
    else:
        locs3D = Location

    
    
    return Feats, Label, Patient_id,locs3D



# @flow(name="Trainning subflow")

def trainning(Images,Label,Patient_id,model,directory,train_part=0.8,batch_size=32,n_epoch=30,n_rep=2,preTrained = False, myArchitectures_pk = None):
    # Introduction: training a simple CNN with the mean of the images.
    # train_part = 0.8
    test_part = 1-train_part

    # batch_size = 32
    # n_epoch = 30
    # n_rep =  2#20
    Mean_Images = np.mean(Images, axis=0)
    Models = []
    print()
    print("Images shape: ", Images.shape)
    print("Mean Image shape: ", Mean_Images.shape)
    print()
    Images = np.transpose(Images, (1, 0, 2, 3, 4)) # (nbExp, n_epoch, nbFreq, xIMG,yIMG)
    nbLabel = np.unique(Label).shape[0]
    if(model == "BasicCNN"):
    
        myShape = Mean_Images.shape
        inputImg = torch.zeros(1, myShape[1], 32, 32)
    else:
        myShape = Images.shape
        inputImg = torch.zeros(1, myShape[1],myShape[2],32,32)
    if not preTrained:
        Model = getattr(mods, model)
        if torch.cuda.is_available():
            Model = Model(input_image=inputImg,n_classes=nbLabel).cuda()
        else:
            Model = Model(input_image=inputImg,n_classes=nbLabel)
    else:
        Model = model
    Model.architecture_pk = myArchitectures_pk
    Result = []
    for r in range(n_rep):
        if(model == "BasicCNN"):
            EEG = EEGImagesDataset(label=Label, image=Mean_Images)
        else:
            print("is not BasiCNN")
            print("Images shape: ", Images.shape)
            print("Label shape: ", Label.shape)
            EEG = EEGImagesDataset(label=Label, image=Images)
        lengths = [int(len(EEG) * train_part), int(len(EEG) * test_part)]
        if sum(lengths) != len(EEG):
            lengths[0] = lengths[0] + 1
            
        Train, Test = random_split(EEG, lengths)
 
        Trainloader = DataLoader(Train, batch_size=batch_size)
        Testloader = DataLoader(Test, batch_size=batch_size)
        try:
            modelTmp,res = TrainTest_Model(Model, Trainloader, Testloader, n_epoch=n_epoch, learning_rate=0.001, print_epoch=-1,
                            opti='Adam')
        except RuntimeError as e:
            print(e)
            continue
        Result.append(res)
        Models.append(modelTmp)
    
    # Result = np.mean(Result, axis=0)
    result_directory = directory+"/Results"
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    sio.savemat(result_directory+"/Res_Basic"+".mat", {"res":Result})
    print ('-'*100)
    print('\nBegin Training for Patient ')
    # print('End Training with \t loss: %.3f\tAccuracy : %.3f\t\tval-loss: %.3f\tval-Accuracy : %.3f' %
    #     (Result[0], Result[1], Result[2], Result[3]))
    print('\n'+'-'*100)

    # Boucle pour sauvegarder chaque modèle dans un fichier pickle
    models_directory = directory + "/Models/"
    if not os.path.exists(models_directory):
        os.makedirs(models_directory)
    
    number = models_directory[models_directory.find("exp")+3:models_directory.find("exp")+4]
    for model in Models:
        filename = model.__class__.__name__ + '_' + number + '.pkl' # Nom du fichier = nom de la classe du modèle
        filepath = os.path.join(models_directory, filename)
        if not os.path.exists(filepath):
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)






# def trainning(Images, Label, Patient_id,model):
#     # Introduction: training a simple CNN with the mean of the images.
#     train_part = 0.8
#     test_part = 0.2

#     batch_size = 32
#     n_epoch = 30
#     n_rep =  2#20
#     Mean_Images = np.mean(Images, axis=0)
#     Models = []
#     print()
#     print("Images shape: ", Images.shape)
#     print()
#     Images = np.transpose(Images, (1, 0, 2, 3, 4)) # (nbExp, n_epoch, nbFreq, xIMG,yIMG)

#     for patient in np.unique(Patient_id):

#         Result = []
#         for r in range(n_rep):
#             if(model == "BasicCNN"):
#                 EEG = EEGImagesDataset(label=Label[Patient_id == patient], image=Mean_Images[Patient_id == patient])
#             else:
#                 print()
#                 print("Images shape: ", Images.shape)
#                 print("Label shape: ", Label.shape)
#                 EEG = EEGImagesDataset(label=Label[Patient_id == patient], image=Images[Patient_id == patient])
#             lengths = [int(len(EEG) * train_part), int(len(EEG) * test_part)]
#             if sum(lengths) != len(EEG):
#                 lengths[0] = lengths[0] + 1
#             Train, Test = random_split(EEG, lengths)
#             Trainloader = DataLoader(Train, batch_size=batch_size)
#             Testloader = DataLoader(Test, batch_size=batch_size)
#             try:
#                 modelTmp,res = TrainTest_Model(getattr(mods, model), Trainloader, Testloader, n_epoch=n_epoch, learning_rate=0.001, print_epoch=-1,
#                                 opti='Adam')
#             except RuntimeError as e:
#                 print(e)
#                 continue
#             Result.append(res)
#             Models.append(modelTmp)
#         sio.savemat("Results/new/Res_Basic_Patient"+str(patient)+".mat", {"res":Result})
#         Result = np.mean(Result, axis=0)
#         print ('-'*100)
#         print('\nBegin Training for Patient '+str(patient))
#         # print('End Training with \t loss: %.3f\tAccuracy : %.3f\t\tval-loss: %.3f\tval-Accuracy : %.3f' %
#         #     (Result[0], Result[1], Result[2], Result[3]))
#         print('\n'+'-'*100)







    # print("\n\n\n\n Maxpool CNN \n\n\n\n")

    # for patient in np.unique(Patient_id):

    #     Result = []
    #     for r in range(n_rep):
    #         EEG = EEGImagesDataset(label=Label[Patient_id == patient], image=Images[Patient_id == patient])
    #         lengths = [int(len(EEG) * train_part + 1), int(len(EEG) * test_part)]
    #         if sum(lengths) < len(EEG):
    #             lengths[0] = lengths[0] + 1
    #         if sum(lengths) > len(EEG):
    #             lengths[0] = lengths[0] - 1
    #         Train, Test = random_split(EEG, lengths)
    #         Trainloader = DataLoader(Train, batch_size=batch_size)
    #         Testloader = DataLoader(Test, batch_size=batch_size)
    #         modelTmp, res = TrainTest_Model(MaxCNN, Trainloader, Testloader, n_epoch=n_epoch, learning_rate=0.001, print_epoch=-1,
    #                             opti='Adam')
    #         Result.append(res)
    #         Models.append(modelTmp)
    #     sio.savemat("Res_MaxCNN_Patient"+str(patient)+".mat", {"res":Result})
    #     Result = np.mean(Result, axis=0)
    #     print ('-'*100)
    #     print('\nBegin Training for Patient '+str(patient))
    #     print('End Training with \t loss: %.3f\tAccuracy : %.3f\t\tval-loss: %.3f\tval-Accuracy : %.3f' %
    #         (Result[0], Result[1], Result[2], Result[3]))
    #     print('\n'+'-'*100)

    # print("\n\n\n\n Temp CNN \n\n\n\n")


    # for patient in np.unique(Patient_id):

    #     Result = []
    #     for r in range(n_rep):
    #         EEG = EEGImagesDataset(label=Label[Patient_id == patient], image=Images[Patient_id == patient])
    #         lengths = [int(len(EEG) * train_part + 1), int(len(EEG) * test_part)]
    #         if sum(lengths) < len(EEG):
    #             lengths[0] = lengths[0] + 1
    #         if sum(lengths) > len(EEG):
    #             lengths[0] = lengths[0] - 1
    #         Train, Test = random_split(EEG, lengths)
    #         Trainloader = DataLoader(Train, batch_size=batch_size)
    #         Testloader = DataLoader(Test, batch_size=batch_size)
    #         modelTmp,res = TrainTest_Model(TempCNN, Trainloader, Testloader, n_epoch=n_epoch, learning_rate=0.001, print_epoch=-1,
    #                             opti='Adam')
    #         Result.append(res)
    #         Models.append(modelTmp)

    #     sio.savemat("Res_TempCNN_Patient"+str(patient)+".mat", {"res":Result})
    #     Result = np.mean(Result, axis=0)
    #     print ('-'*100)
    #     print('\nBegin Training for Patient '+str(patient))
    #     print('End Training with \t loss: %.3f\tAccuracy : %.3f\t\tval-loss: %.3f\tval-Accuracy : %.3f' %
    #         (Result[0], Result[1], Result[2], Result[3]))
    #     print('\n'+'-'*100)


    # print("\n\n\n\n LSTM CNN \n\n\n\n")


    # for patient in np.unique(Patient_id):

    #     Result = []
    #     for r in range(n_rep):
    #         EEG = EEGImagesDataset(label=Label[Patient_id == patient], image=Images[Patient_id == patient])
    #         lengths = [int(len(EEG) * train_part + 1), int(len(EEG) * test_part)]
    #         if sum(lengths) < len(EEG):
    #             lengths[0] = lengths[0] + 1
    #         if sum(lengths) > len(EEG):
    #             lengths[0] = lengths[0] - 1
    #         Train, Test = random_split(EEG, lengths)
    #         Trainloader = DataLoader(Train, batch_size=batch_size)
    #         Testloader = DataLoader(Test, batch_size=batch_size)
    #         modelTmp,res = TrainTest_Model(LSTM, Trainloader, Testloader, n_epoch=n_epoch, learning_rate=0.001, print_epoch=1,
    #                             opti='Adam')
    #         Result.append(res)
    #         Models.append(modelTmp)
    #     sio.savemat("Res_LSTM_Patient"+str(patient)+".mat", {"res":Result})
    #     Result = np.mean(Result, axis=0)
    #     print ('-'*100)
    #     print('\nBegin Training for Patient '+str(patient))
    #     print('End Training with \t loss: %.3f\tAccuracy : %.3f\t\tval-loss: %.3f\tval-Accuracy : %.3f' %
    #         (Result[0], Result[1], Result[2], Result[3]))
    #     print('\n'+'-'*100)


    # print("\n\n\n\n Mix CNN \n\n\n\n")


    # for patient in np.unique(Patient_id):

    #     Result = []
    #     for r in range(n_rep):
    #         EEG = EEGImagesDataset(label=Label[Patient_id == patient], image=Images[Patient_id == patient])
    #         lengths = [int(len(EEG) * train_part + 1), int(len(EEG) * test_part)]
    #         if sum(lengths) < len(EEG):
    #             lengths[0] = lengths[0] + 1
    #         if sum(lengths) > len(EEG):
    #             lengths[0] = lengths[0] - 1
    #         Train, Test = random_split(EEG, lengths)
    #         Trainloader = DataLoader(Train, batch_size=batch_size)
    #         Testloader = DataLoader(Test, batch_size=batch_size)
    #         modelTmp,res = TrainTest_Model(Mix, Trainloader, Testloader, n_epoch=n_epoch, learning_rate=0.001, print_epoch=-1,
    #                             opti='Adam')
    #         Result.append(res)
    #         Models.append(modelTmp)
    #     sio.savemat("Res_Mix_Patient"+str(patient)+".mat", {"res":Result})
    #     Result = np.mean(Result, axis=0)
    #     print ('-'*100)
    #     print('\nBegin Training for Patient '+str(patient))
    #     print('End Training with \t loss: %.3f\tAccuracy : %.3f\t\tval-loss: %.3f\tval-Accuracy : %.3f' %
    #         (Result[0], Result[1], Result[2], Result[3]))
    #     print('\n'+'-'*100)
        
    #     return Models