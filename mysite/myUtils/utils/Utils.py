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
import io
import os
from django.conf import settings
import mne
import glob
from scipy import signal
import pandas as pd
from torch.utils.data.dataset import Dataset
from .Utils_Bashivan import *

import torch

import scipy.io as sio
import torch.optim as optim
import torch.nn as nn
import numpy as np
# from prefect import flow, task



def kfold(length, n_fold):
    tot_id = np.arange(length)
    np.random.shuffle(tot_id)
    len_fold = int(length/n_fold)
    train_id = []
    test_id = []
    for i in range(n_fold):
        test_id.append(tot_id[i*len_fold:(i+1)*len_fold])
        train_id.append(np.hstack([tot_id[0:i*len_fold],tot_id[(i+1)*len_fold:-1]]))
    return train_id, test_id


class EEGImagesDataset(Dataset):
    """EEGLearn Images Dataset from EEG."""
    
    def __init__(self, label, image):
        self.label = label
        self.Images = image
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.Images[idx]
        label = self.label[idx]
        sample = (image, label)
        
        return sample

# @task(name="test model", log_prints=False)
def Test_Model(net, Testloader, criterion, is_cuda=True):
    # running_loss = 0.0 
    # evaluation = []
    # for i, data in enumerate(Testloader, 0):
    #     input_img, labels = data
    #     input_img = input_img.to(torch.float32)
    #     if is_cuda:
    #         input_img = input_img.cuda()
    #     outputs = net(input_img)
    #     _, predicted = torch.max(outputs.cpu().data, 1)
    #     evaluation.append((predicted==labels).tolist())
    #     loss = criterion(outputs, labels.cuda())
    #     running_loss += loss.item()
    # running_loss = running_loss/(i+1)
    # evaluation = [item for sublist in evaluation for item in sublist]
    # running_acc = sum(evaluation)/len(evaluation)
    # return running_loss, running_acc
    running_loss = 0.0 
    evaluation = []
    for i, data in enumerate(Testloader, 0):
        input_img, labels = data
        input_img = input_img.to(torch.float32)
        if is_cuda:
            input_img = input_img.cuda()
        outputs = net(input_img)
        _, predicted = torch.max(outputs.cpu().data, 1)
        evaluation.append((predicted==labels).tolist())
        loss = criterion(outputs, labels.type(torch.LongTensor).cuda())
        running_loss += loss.item()
    running_loss = running_loss/(i+1)
    evaluation = [item for sublist in evaluation for item in sublist]
    running_acc = sum(evaluation)/len(evaluation)
    return running_loss, running_acc


# @flow(name="train test model subflow")
def TrainTest_Model(model, trainloader, testloader, n_epoch=30, opti='SGD', learning_rate=0.0001, is_cuda=True, print_epoch =5, verbose=False):
    net = model
        
    criterion = nn.CrossEntropyLoss()
    
    if opti=='SGD':
        optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    elif opti =='Adam':
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    else: 
        print("Optimizer: "+optim+" not implemented.")
    
    for epoch in range(n_epoch):
        running_loss = 0.0
        evaluation = []
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.to(torch.float32).cuda())
            _, predicted = torch.max(outputs.cpu().data, 1)
            evaluation.append((predicted==labels).tolist())
            loss = criterion(outputs, labels.cuda().long())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        running_loss = running_loss/(i+1)
        evaluation = [item for sublist in evaluation for item in sublist]
        running_acc = sum(evaluation)/len(evaluation)
        validation_loss, validation_acc = Test_Model(net, testloader, criterion,True)
        
        if epoch%print_epoch==(print_epoch-1):
            print('[%d, %3d]\tloss: %.3f\tAccuracy : %.3f\t\tval-loss: %.3f\tval-Accuracy : %.3f' %
             (epoch+1, n_epoch, running_loss, running_acc, validation_loss, validation_acc))
    if verbose:
        print('Finished Training \n loss: %.3f\tAccuracy : %.3f\t\tval-loss: %.3f\tval-Accuracy : %.3f' %
                 (running_loss, running_acc, validation_loss,validation_acc))
    
    return net,(running_loss, running_acc, validation_loss,validation_acc)


# def create_img(directory=None,feats=None,locs=None):
    
    
    
#     if(feats is None):
#         feats = sio.loadmat('Sample Data/FeatureMat_timeWin.mat')['features']
#     if(locs is None):
#         locs = sio.loadmat('Sample Data/Neuroscan_locs_orig.mat')
#         locs_3d = locs['A']
#     else:
#         locs_3d = locs
#     locs_2d = []
#     # Convert to 2D
#     for e in locs_3d:
#         locs_2d.append(azim_proj(e))

#     images_timewin = np.array([gen_images(np.array(locs_2d),
#                                           feats[:, i * 192:(i + 1) * 192], 32, normalize=True) for i in
#                                range(int(feats.shape[1] / 192))
#                                ])

#     # saveAs = "Sample Data/images_time.mat"
#     if(directory is None):
#         directory = "dataSet/images_time.mat"
#     saveAs = directory
#     sio.savemat(saveAs,{"img":images_timewin})
#     print("Images Created and Save in Sample Dat/images_time")
#     # return {"img":images_timewin}
#     return images_timewin

def create_img(feats, locs, n_freq, n_epoch, img_size = 32, directory=None):
    """
    Génère des images à partir des caractéristiques extraites des données d'EEG.

    :param feats: Matrice de caractéristiques de taille (n_samples, n_channels * n_freq * n_epoch + 1).
    :param locs: Matrice de localisation de capteurs de taille (n_channels, 3).
    :param n_freq: Nombre de fréquences par instant dans le temps.
    :param n_epoch: Nombre d'épochs dans les données.
    :param img_size: Taille des images à générer (un entier pour la hauteur et la largeur).
    :param directory: Optionnel, répertoire de sauvegarde pour les images.
    :return: Matrice d'images de taille (n_samples, n_epoch, n_freq, img_size, img_size).
    """
    n_freq = int(n_freq)
    n_epoch = int(n_epoch)
    img_size = int(img_size)
    feats = feats[:, :-1] # On enlève la dernière colonne qui contient les labels

    n_channels = int(feats.shape[1] / (n_freq * n_epoch ))
    locs_3d = locs
    locs_2d = [azim_proj(e) for e in locs_3d]
    images = np.array([
        gen_images(np.array(locs_2d), feats[:, i * n_channels * n_freq:(i + 1) * n_channels * n_freq], img_size,
                   normalize=True)
        for i in range(n_epoch)
    ])

    if directory is None:
        directory = "dataSet/images_time.mat"
    saveAs = directory
    sio.savemat(saveAs, {"img": images})
    print(f"Images created and saved in {directory}")
    return images


def extract_frequency_power(csv_file,electrodes, num_epochs, frequencies):
    # Charger les données du fichier CSV
    # data = pd.read_csv(csv_file)
    # print("csv_file",csv_file)
    with open(csv_file.path, 'r') as f:
        data = pd.read_csv(f,delimiter=',') 
        columns_to_drop = [col for col in data.columns if col not in electrodes]
        data = data.drop(columns_to_drop, axis=1)
    # Obtenir le nombre total de capteurs
    num_sensors = len(data.columns)
    
    # Calculer la durée d'un epoch
    epoch_duration = int(len(data.index) / num_epochs)
    
    # Initialiser la matrice de puissance spectrale pour chaque epoch et chaque capteur
    power_matrix = np.zeros((num_epochs, num_sensors, len(frequencies)))
    
    # Boucle à travers chaque capteur
    for sensor in range(num_sensors):
        # Obtenir les données pour le capteur courant
        sensor_data = data.iloc[:, sensor]
        
        # Boucle à travers chaque epoch
        for epoch in range(num_epochs):
            # Obtenir les données pour l'epoch courant
            epoch_data = sensor_data[epoch*epoch_duration:(epoch+1)*epoch_duration]
            
            # Appliquer la transformation de Fourier à court terme pour obtenir la puissance spectrale
            f, Pxx = signal.periodogram(epoch_data, fs=250, window='hamming', scaling='spectrum')
            
            # Initialiser la liste de quantités d'énergie pour les fréquences demandées
            energy = []
            for freq in frequencies:
                if freq == 'gamma':
                    energy.append(np.sum(Pxx[(f >= 30) & (f <= 100)]))
                elif freq == 'beta':
                    energy.append(np.sum(Pxx[(f >= 13) & (f < 30)]))
                elif freq == 'alpha':
                    energy.append(np.sum(Pxx[(f >= 8) & (f <= 13)]))
                elif freq == 'theta':
                    energy.append(np.sum(Pxx[(f >= 4) & (f < 8)]))
                elif freq == 'delta':
                    energy.append(np.sum(Pxx[(f >= 0.5) & (f < 4)]))
                else:
                    raise ValueError(f"Invalid frequency '{freq}'.")
            
            # Calculer la moyenne des quantités d'énergie pour les fréquences demandées
            mean_energy = np.mean(energy)
            
            # Enregistrer la moyenne des quantités d'énergie dans la matrice de puissance spectrale
            power_matrix[epoch, sensor, :] = energy
    
    # Aplatir la matrice de puissance spectrale pour obtenir une liste de toutes les mesures
    power_list = power_matrix.reshape(-1, len(frequencies))
    
    # Retourner la liste de puissance spectrale
    return power_list


# @task(name="locations_toMatrice", log_prints=True)
def locationsToMatrice(path):
    """
    Convertie le fichier csv des localisations des capteurs en matrice
    :param path: Chemin vers le fichier csv
    :return: matrice des localisations des capteurs
    """
    df = pd.read_csv(path)
    try:
        os.makedirs(path[:-13]+'mat')
        print("Répertoire créé avec succès.")
    except OSError as e:
        print(f"Erreur lors de la création du répertoire : {e}")
    sio.savemat(path[:-13]+'mat/locs.mat', {'locs': df.values})
    test = sio.loadmat(path[:-13]+'mat/locs.mat')["locs"]
    return test

# @task(name="patient_id_toMatrice", log_prints=True)
def patientIdToMatrice(file):
    """
    Convertie le fichier csv des identifiants des patients en matrice
    :param path: Chemin vers le fichier csv
    :return: matrice des identifiants des patients
    """
    df = pd.read_csv(io.BytesIO(file))

    # df = pd.read_csv(path)
    data = np.array(df[df.columns[0]])
    sio.savemat('dataSet/trials_subNums.mat', {'subjectNum': data})
    test = sio.loadmat('dataSet/trials_subNums.mat')["subjectNum"][0]#corresponding to the signal features
    return test

# @task(name="getMontage")
def getMontage(wd,user,montage="standard_1005",col_names=None):
    # col_names = col_names.split(",")
    montage = mne.channels.make_standard_montage(montage)
    # Obtenir les positions x, y et z de toutes les électrodes
    positions = montage.get_positions()
    myDataFrame = pd.DataFrame(positions['ch_pos'].values(), columns=['x', 'y', 'z'])
    myDataFrame['caps'] = positions['ch_pos'].keys()
    #reorganiser l'ordre de mes colonnes
    myDataFrame = myDataFrame[['caps', 'x', 'y', 'z']]
    df_filtre = myDataFrame[myDataFrame['caps'].isin(col_names)]

    # Enregistrer le dataframe dans un fichier csv
    dataFrameLoc = df_filtre[['x', 'y', 'z']]
    path = settings.MEDIA_ROOT+'/uploads/'+str(user.username)+f'/exp{wd.numExp}'+'/locations.csv'
    dataFrameLoc.to_csv(path, index=False)
    # ? Il faudrait voir si on retourne la matrice sans nom ou le dataframe pour matcher les capteurs voulu
    return locationsToMatrice(path)

def getMontageList():
    myList = [
        {"standard_1005":"Electrodes are named and positioned according to the international 10-05 system (343+3 locations)"},
        {"standard_1020":"Electrodes are named and positioned according to the international 10-20 system (94+3 locations)"},
        {"standard_alphabetic": "Electrodes are named with LETTER-NUMBER combinations (A1, B2, F4, …) (65+3 locations)"},
        {"standard_postfixed": "Electrodes are named according to the international 10-20 system using postfixes for intermediate positions (100+3 locations)"},
        {"standard_prefixed" : "Electrodes are named according to the international 10-20 system using prefixes for intermediate positions (74+3 locations)"},
        {"standard_primed": "Electrodes are named according to the international 10-20 system using prime marks (' and '') for intermediate positions (100+3 locations)"},
        {"biosemi16": "BioSemi cap with 16 electrodes (16+3 locations)"},
        {"biosemi32": "BioSemi cap with 32 electrodes (32+3 locations)"},
        {"biosemi64": "BioSemi cap with 64 electrodes (64+3 locations)"},
        {"biosemi128": "BioSemi cap with 128 electrodes (128+3 locations)"},
        {"biosemi160": "BioSemi cap with 160 electrodes (160+3 locations)"},
        {"biosemi256": "BioSemi cap with 256 electrodes (256+3 locations)"},
        {"easycap-M1": "EasyCap with 10-05 electrode names (74 locations)"},
        {"easycap-M10": "EasyCap with numbered electrodes (61 locations)"},
        {"EGI_256": "Geodesic Sensor Net (256 locations)"},
        {"GSN-HydroCel-32": "HydroCel Geodesic Sensor Net and Cz (33+3 locations)"},
        {"GSN-HydroCel-32_1.0": "HydroCel Geodesic Sensor Net (32+3 locations)"},
        {"GSN-HydroCel-64_1.0": "HydroCel Geodesic Sensor Net (64+3 locations)"},
        {"GSN-HydroCel-65_1.0": "HydroCel Geodesic Sensor Net and Cz (65+3 locations)"},
        {"GSN-HydroCel-128": "HydroCel Geodesic Sensor Net (128+3 locations)"},
        {"GSN-HydroCel-129": "HydroCel Geodesic Sensor Net and Cz (129+3 locations)"},
        {"GSN-HydroCel-256": "HydroCel Geodesic Sensor Net (256+3 locations)"},
        {"GSN-HydroCel-257": "HydroCel Geodesic Sensor Net and Cz (257+3 locations)"},
        {"mgh60": "The (older) 60-channel cap used at MGH (60+3 locations)"},
        {"mgh70": "The (newer) 70-channel BrainVision cap used at MGH (70+3 locations)"},
        {"artinis-octamon": "Artinis OctaMon fNIRS (8 sources, 16 detectors)"},
        {"artinis-brite23": "Artinis Brite23 fNIRS (11 sources, 7 detectors)"},
        {"brainproducts-RNP-BA-128": "Brain Products with 10-10 electrode names (128 channels)"},
    ]
    return myList

# @task(name="to matrice", log_prints=True)
def convertCSVsTOmat(files,labels,path,electrodes,epoch=10,frequencies=["beta","gamma","alpha","theta","delta"]):
    """
    Convertie les fichiers csv en matrice de puissance spectrale
    prend en paramètre le chemin vers les fichiers csv qui doivent être dans le format suivant :
        - 1 colonne par capteur
        - 1 ligne par échantillon
        - 1 fichier par prelevement
        - 1 fichier par patient
    avec le nom du fichier de la forme : "nomFichier label.csv"
    :param path: Chemin vers les fichiers csv
    :return: Matrice de puissance spectrale
    """
    # files = glob.glob(path)
    # epoch = 10  # Nombre d'échantillons par epoch
    # freqs = ["all"]  # Fréquences calculées
    epoch = int(epoch)
    # first_f = pd.read_csv(files[0],delimiter=',') 
    # with open(files[0].path, 'r') as f:
    #     df = pd.read_csv(f, delimiter=',')
        # Effectuer vos opérations de lecture ou de traitement sur df
    num_sensors = len(electrodes)  # Nombre de capteurs EEG
    print("num_sensors",num_sensors)
    num_files = len(files)
    with open(labels.path, 'r') as f:
        labels = pd.read_csv(f,delimiter=',')
    # print("labels ",labels)

    # frequencies = ["beta","gamma","alpha","theta","delta"]
    data_mat = np.zeros((num_files, epoch*num_sensors*len(frequencies) +1))# Nbfiles or prelevments , 10 epochs * 4 capteurs *5 fréquences + 1 label
    for i in range(num_files):
        
        power = extract_frequency_power(files[i],electrodes, epoch, frequencies)
        data_mat[i, :-1] = np.reshape(power, -1)
        try:
            # print("labels['Label'][i]",labels['Label'][i])
            data_mat[i, -1] = int(labels['Label'][i])
        except:
            data_mat[i, -1] = 000
    sio.savemat(os.path.join(path, "donnees_eeg.mat"), {"data": data_mat})
    return data_mat