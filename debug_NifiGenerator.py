from utils import NiftiGenerator
import os
import glob
import numpy as np


def split_dataset(folderX, folderY, validation_ratio):

    train_folderX = folderX + "/trainX/"
    train_folderY = folderY + "/trainY/"
    valid_folderX = folderX + "/validX/"
    valid_folderY = folderY + "/validY/"

    if not os.path.exists(train_folderX):
        os.makedirs(train_folderX)
    if not os.path.exists(train_folderY):
        os.makedirs(train_folderY)
    if not os.path.exists(valid_folderX):
        os.makedirs(valid_folderX)
    if not os.path.exists(valid_folderY):
        os.makedirs(valid_folderY)

    # data_trainX_list.sort()
    # data_validX_list.sort()
    # data_trainY_list.sort()
    # data_validY_list.sort()

    # print(data_trainX_list)
    # print(data_validX_list)
    # print(data_trainY_list)
    # print(data_validY_list)

    data_path_list = glob.glob(folderX+"/*.nii") + glob.glob(folderX+"/*.nii.gz")
    data_path_list.sort()
    print(data_path_list)
    data_path_list = np.asarray(data_path_list)
    np.random.shuffle(data_path_list)
    data_path_list = list(data_path_list)
    data_name_list = []
    for data_path in data_path_list:
        data_name_list.append(os.path.basename(data_path))

    valid_list = data_name_list[:int(len(data_name_list)*validation_ratio)]
    valid_list.sort()
    train_list = list(set(data_name_list) - set(valid_list))
    train_list.sort()

    print("valid_list: ", valid_list)
    print('-'*50)
    print("train_list: ", train_list)

    for valid_name in valid_list:
        valid_nameX = folderX+"/"+valid_name
        valid_nameY = folderY+"/"+valid_name.replace("NACB", "CTAC")
        cmdX = "mv "+valid_nameX+" "+valid_folderX
        cmdY = "mv "+valid_nameY+" "+valid_folderY
        print(cmdX)
        print(cmdY)
        os.system(cmdX)
        os.system(cmdY)

    for train_name in train_list:
        train_nameX = folderX+"/"+train_name
        train_nameY = folderY+"/"+train_name.replace("NACB", "CTAC")
        cmdX = "mv "+train_nameX+" "+train_folderX
        cmdY = "mv "+train_nameY+" "+train_folderY
        print(cmdX)
        print(cmdY)
        os.system(cmdX)
        os.system(cmdY)

    return [train_folderX, train_folderY, valid_folderX, valid_folderY]


para_name = "ex01"
# Data to be written  
train_para ={  
    "para_name" : para_name,
    "img_rows" : 512, # image is resampled to this size
    "img_cols" : 512, # image is resampled to this size
    "channel_X" : 5,
    "channel_Y" : 1,
    "start_ch" : 32,
    "depth" : 3, 
    "validation_split" : 0.2,
    "loss" : "l1",
    "x_data_folder" : 'NAC',
    "y_data_folder" : 'CTAC',
    "weightfile_name" : 'weights_'+para_name+'.h5',
    "model_name" : 'model_'+para_name+'.json',
    "save_folder" : './achives/',
    "jpgprogressfile_name" : 'progress_'+para_name,
    "batch_size" : 4, # should be smallish. 1-10
    "num_epochs" : 25, # should train for at least 100-200 in total
    "steps_per_epoch" : 20*89, # should be enough to be equal to one whole pass through the dataset
    "initial_epoch" : 0, # for resuming training
    "load_weights" : False, # load trained weights for resuming training
}



print('-'*50)
print('Setting up NiftiGenerator')
print('-'*50)
niftiGen_augment_opts = NiftiGenerator.PairedNiftiGenerator.get_default_augOptions()
niftiGen_augment_opts.hflips = True
niftiGen_augment_opts.vflips = True
niftiGen_augment_opts.rotations = 15
niftiGen_augment_opts.scalings = 0
niftiGen_augment_opts.shears = 0
niftiGen_augment_opts.translations = 10
print(niftiGen_augment_opts)
niftiGen_norm_opts = NiftiGenerator.PairedNiftiGenerator.get_default_normOptions()
niftiGen_norm_opts.normXtype = 'auto'
niftiGen_norm_opts.normYtype = 'auto'
print(niftiGen_norm_opts)

folderX = "./data_train/"+train_para["x_data_folder"]
folderY = "./data_train/"+train_para["y_data_folder"]
folder_list = [folderX, folderY]
sub_folder_list = split_dataset(folderX=folderX, folderY=folderY, 
                                validation_ratio=train_para["validation_split"])
[train_folderX, train_folderY, valid_folderX, valid_folderY] = sub_folder_list
print(train_folderX, train_folderY, valid_folderX, valid_folderY)

niftiGenT = NiftiGenerator.PairedNiftiGenerator()
niftiGenT.initialize(train_folderX, train_folderY,
                     niftiGen_augment_opts, niftiGen_norm_opts)
generatorT = niftiGenT.generate(img_size=(train_para["img_rows"],train_para["img_cols"]),
                                Xslice_samples=train_para["channel_X"],
                                Yslice_samples=train_para["channel_Y"],
                                batch_size=train_para["batch_size"])

niftiGenV = NiftiGenerator.PairedNiftiGenerator()
niftiGenV.initialize(valid_folderX, valid_folderY,
                     niftiGen_augment_opts, niftiGen_norm_opts )
generatorV = niftiGenV.generate(img_size=(train_para["img_rows"],train_para["img_cols"]),
                                Xslice_samples=train_para["channel_X"],
                                Yslice_samples=train_para["channel_Y"],
                                batch_size=train_para["batch_size"])


for [batch_X , batch_Y] in generatorT:
    print(batch_X.shape, batch_Y.shape)
    np.save("batch_X_genT.npy", batch_X)
    np.save("batch_Y_genT.npy", batch_Y)
    exit()

