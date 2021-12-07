import os
import pandas as pd
from sklearn.utils import resample
from glob import glob
from sklearn.model_selection import train_test_split
from PIL import Image



def rotate_images(input_file_locations):
    # Figue out which ones we are trying to classify
    emotions = glob(input_file_locations + "*")

    for emotion in emotions:
        files = glob(emotion+"/*")
        for file in files:
            im = Image.open(file)
            directory = file.split('/')[0:2]
            name = file.split('/')[2].split('.')[0]
            for degree in [90, 180, 270]:
                im=im.rotate(90, expand=True)
                im.save(f"{directory[0]}/{directory[1]}/{name}_{degree}.jpg")


def prepare_data(input_file_locations, traindir, validdir, testdir,do_oversampling, oversampling):
    
    # (Re)create folder structure
    os.system(f"rm -rf data/")
    os.system(f"mkdir data")
    os.system(f"rm -rf data/")
    os.system(f"rm -rf data/train")
    os.system(f"rm -rf data/test")
    os.system(f"rm -rf data/val")
    os.system(f"mkdir data/")
    os.system(f"mkdir data/train")
    os.system(f"mkdir data/test")
    os.system(f"mkdir data/val")
    
    # Might be redundant leftover
    os.system(f"mkdir outputs")
    
    # Figue out which ones we are trying to classify
    emotions = glob(input_file_locations + "*")

    # Do stuff
    filedf = pd.DataFrame()
    for emotion in emotions:
        files = glob(emotion+"/*")
        tempdf = pd.DataFrame({'filepath':files,'category':emotion.split("/")[-1]})
        filedf = pd.concat([filedf,tempdf])
        
    # Split in stratified way
    X_train, X_test, _, _ = train_test_split(
            filedf, filedf['category'],stratify=filedf['category'], test_size=0.4)

    X_test, X_val, _, _ = train_test_split(
            X_test, X_test['category'], stratify=X_test['category'], test_size=0.5)
    
    
    X_train['type'] = 'train'
    X_val['type'] = 'val'
    X_test['type'] = 'test'

    if do_oversampling:
        # Resample/oversample due to class imbalance
        amount_to_sample = X_train.groupby("category")\
                        .aggregate('count')\
                        .rename(columns = {'filepath':'cnt'})\
                        .reset_index()\
                        .sort_values(by='cnt',ascending=False).cnt.max()

        amount_to_sample = amount_to_sample*oversampling
        amount_to_sample

        # Oversample
        X_train_resampled = X_train.head(0).copy()
        for em in X_train.category.unique():
            oversampl = resample(X_train[X_train.category == em],replace=True , n_samples=amount_to_sample, random_state=42)
            X_train_resampled = pd.concat([X_train_resampled, oversampl])

        # Concatenate
        fulldf = pd.concat([X_train_resampled,X_test,X_val])
    else: # If no oversampling
        fulldf = pd.concat([X_train,X_test,X_val])

    # Create base folders
    for cat in fulldf.category.unique():
        os.system(f"mkdir data/train/'{cat}'") 
        os.system(f"mkdir data/test/'{cat}'") 
        os.system(f"mkdir data/val/'{cat}'")


    # Copy files
    for i,row in fulldf.iterrows():
        cat = row['category']
        section = row['type']
        ipath = row['filepath']
        # output filepath to paste
        opath = ipath.replace(f"images_final/",f"data/{section}/")
        opath = opath.replace(f".jpg",f"_{i}.jpg")
        #print(f"cp '{ipath}' '{opath}'")
        os.system(f"cp '{ipath}' '{opath}'")