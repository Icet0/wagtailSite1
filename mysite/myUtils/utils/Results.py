import io
import scipy.io as sio
import matplotlib.pyplot as plt

import os
import glob

import numpy as np
from prefect import flow, task

@task
def result(dir):
    try:
        models = ["Basic"]#, "LSTM", "MaxCNN", "Mix", "TempCNN"]

        # Initialisation des listes de performances pour chaque modèle
        running_losses = []
        running_accs = []
        validation_losses = []
        validation_accs = []

        for model in models:
            # Chargement des données pour le modèle courant
            doc = glob.glob(dir + "\\*" + model + ".mat")[0]
            file = sio.loadmat(doc)['res']
            
            # Récupération des performances pour chaque répétition
            running_loss = file[:, 0]
            running_acc = file[:, 1]
            validation_loss = file[:, 2]
            validation_acc = file[:, 3]
            
            # Ajout des performances moyennes à la liste correspondante
            running_losses.append(np.mean(running_loss))
            running_accs.append(np.mean(running_acc))
            validation_losses.append(np.mean(validation_loss))
            validation_accs.append(np.mean(validation_acc))
            
        # Affichage des performances moyennes dans un graphique
        x = np.arange(len(models))
        width = 0.2
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width, running_losses, width, label='Running Loss')
        rects2 = ax.bar(x, running_accs, width, label='Running Accuracy')
        rects3 = ax.bar(x + width, validation_losses, width, label='Validation Loss')
        rects4 = ax.bar(x + 2 * width, validation_accs, width, label='Validation Accuracy')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        # plt.show()

        # plt.show()
        plt.savefig(dir+"perf.png")
        
        # Convert plot to byte string
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format='png')
        img_bytes.seek(0)
        # Return byte string as response
        plt.close()

        return img_bytes.getvalue()
    except Exception as e:
        return "Error, no data found for obtaining the results."





# def result(Patient_id,dir):
#     try:
#         Patient_id = np.unique(Patient_id) #corresponding to the patient id
#         models = ["Basic", "LSTM", "MaxCNN", "Mix", "TempCNN"]

#         Results = np.zeros((len(Patient_id), len(models),20))

#         inc = 0
#         for model in models:
#             inc_patient = 0
#             for patient in Patient_id:
#                 doc = glob.glob(dir+"*"+model+"*"+"t"+str(patient)+".mat")[0]
#                 file = sio.loadmat(doc)['res']
#                 Results[inc_patient,inc, :] = file[:,3]
#                 inc_patient += 1
#             inc += 1

#         fig = plt.figure()

#         for i in range(len(models)):
#             a = 5
#             plt.plot(np.max(Results[:,i,:],axis=1), '.-', label = models[i])

#         plt.legend()
#         #plt.boxplot(Results[:,0,:])
#         #plt.boxplot(Results[:,1,:])



#         lstm = sio.loadmat(dir+"result_LSTM.mat")['vacc']
#         plt.plot(np.mean(lstm, axis=0))

#         # plt.show()
#         plt.savefig(dir+"perf.png")
        
#         # Convert plot to byte string
#         img_bytes = io.BytesIO()
#         plt.savefig(img_bytes, format='png')
#         img_bytes.seek(0)
#         # Return byte string as response
#         plt.close()

#         return img_bytes.getvalue()
#     except:
#         return "Error, no data found for obtaining the results."