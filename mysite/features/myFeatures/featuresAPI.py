import os
import pandas as pd
from .features_calculs import *
import numpy as np



#un mapping de mon api qui permet de récupérer un lien dun csv dans mon body en python flask?
def addFeatures(csv,names=None,epoch=1):

    if(not names):
        names = None
    # else:
    #     names = names
    print("names",names)
    
    if(csv):
        df = pd.read_csv(csv)
        # sensors = ['TP9', 'AF7', 'AF8', 'TP10']
        sensors = df.columns[:]
        try:
            features = calculate_features(df,names)
            # df = pd.DataFrame.from_dict(features)
            # concatenate dataframes in the dict
            # for key in features.keys():
            #     features[key] = pd.concat([features[key], pd.DataFrame(eval(key))], axis=0, ignore_index=True)
            # Concaténer les DataFrames dans un DataFrame unique
            df = pd.DataFrame(columns=features.keys(), index=sensors)
            
            # Remplissage du DataFrame avec les données du dictionnaire
            for key, values in features.items():
                for i in range (values.shape[1]):
                    df[key][i] = values[i][0]
            print(df)
            # print(features)
        except Exception as e:
            return str(e)
    # Obtenir le répertoire parent du chemin
    parent_directory = os.path.dirname(csv)

    # Enlever le dernier "/"
    parent_directory = parent_directory.rstrip("/")

    # Former le nouveau chemin avec le dossier "Results"
    new_path = os.path.join(parent_directory, "Results")
    #vérifier que path existe sinon creer le dossier
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    df.to_csv(new_path+"/myFeatures.csv",index_label='Sensors')    
    return new_path+"/myFeatures.csv"



#Functions 
def calculate_features(df,names=None):    
    fs = 256
    eegData = np.array(df[df.columns[:]]) # remove the first column (time)
    print(eegData)
    [nt, nc] = eegData.shape
    print('EEG shape = ({} timepoints, {} channels)'.format(nt, nc))
    
    lvl = defineEEGFreqs()
    numChans = eegData.shape[1]
    print('Number of channels = {}'.format(numChans))
    functions = { 'shannon entropy': 'calcShannonEntropy(epoch, lvl, nt, nc, fs)'
                 , 'spectral edge frequency': 'calcSpectralEdgeFreq(epoch, lvl, nt, nc, fs)'
                 , 'correlation matrix (channel)' : 'calcCorrelationMatrixChan(epoch)'
                 , 'correlation matrix (frequency)' : 'calcCorrelationMatrixFreq(epoch, lvl, nt, nc, fs)'
                 , 'shannon entropy (dyad)' : 'calcShannonEntropyDyad(epoch, lvl, nt, nc, fs)'
                 , 'crosscorrelation (dyad)' : 'calcXCorrChannelsDyad(epoch, lvl, nt, nc, fs)'
                 , 'hjorth activity' : 'calcActivity(epoch)'
                 , 'hjorth mobility' : 'calcMobility(epoch)'
                 , 'hjorth complexity' : 'calcComplexity(epoch)'
                 , 'skewness' : 'calcSkewness(epoch)'
                 , 'kurtosis' : 'calcKurtosis(epoch)'
                 , 'Petrosian FD' : 'calcPetrosianFD(epoch)'
                 , 'Hjorth FD' : 'calcHjorthFD(epoch)'
                 , 'Katz FD' : 'calcKatzFD(epoch)'
                 , 'Higuchi FD' : 'calcHiguchiFD(epoch)'
                 , 'calcSampleEntropy' : 'calcSampleEntropy(epoch)'   #LONGUEEEEE
                , 'calcWE' : 'calcWE(epoch)'
                , 'calsSE' : 'calcSE(epoch)'
                , 'calcSPEn' : 'calcSPEn(epoch,fs)'
                , 'calc_PP_SampEn' : 'calc_PP_SampEn(epoch)'
                , 'calcApEn' : 'calcApEn(epoch)'
                , 'calcTWE' : 'calcTWE(epoch,fs)'
                # , 'calcWaveletTransform' : 'calcWaveletTransform(epoch)'
                # , 'Detrended Fluctuation Analysis' : 'calcDFA(epoch)'  # DFA takes a long time!
                 }
    
    
    keyName = functions.items()
    myKey = []
    for i in keyName:
        if i[0] in names:
            myKey.append(i)
    # Initialize a dictionary of pandas dataframes with the features as keys
    if(names != None):
        feat = {key[0]: pd.DataFrame() for key in myKey}  
    else:
        feat = {key[0]: pd.DataFrame() for key in functions.items()}  


    epoch = eegData
    
    if(names != None):
        for key in myKey:
            # feat[key[0]] = feat[key[0]].append(pd.DataFrame(eval(key[1])).T)
            feat[key[0]] = pd.concat([feat[key[0]], pd.DataFrame(eval(key[1])).T], axis=0, ignore_index=True)
    else:
        for key in functions.items():
            # feat[key[0]] = feat[key[0]].append(pd.DataFrame(eval(key[1])).T)
            feat[key[0]] = pd.concat([feat[key[0]], pd.DataFrame(eval(key[1])).T], axis=0, ignore_index=True)

             
   
    
    return feat


