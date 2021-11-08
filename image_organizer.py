import shutil, os
import pathlib
import pandas as pd

#Run this command to download dataset from Kaggle.
#kaggle datasets download -d arashnic/faces-age-detection-dataset

# Read the csv data file with pandas.
data_csv = pd.read_csv('/Users/cadentait/python/github portfolio/Age Detection/train.csv')

# Get labels of classes.
labels = data_csv.sort_values('Class')

# Make a list of class names.
class_names = list(labels.Class.unique())

# Path to folder with downloaded images that aren't sorted by class.
age_detection_folder = '/Users/cadentait/python/github portfolio/Age Detection/'

# Create new folders based on class.
for i in class_names:
    try:
        os.makedirs(os.path.join(age_detection_folder, 'Train_', i))
        print('Made path of', str(os.path.join(age_detection_folder, 'Train_', i)))
    except FileExistsError:
        print('THE FILE EXISTS ALREADY...MOVING ON')

# Place images into their correct class folders.
for c in class_names:
    list_of_images = list(labels[labels['Class']==c]['ID'])
    for i in list_of_images:
        try:
            get_image = os.path.join('/Users/cadentait/python/github portfolio/Age Detection/Train/'+i)
            
            move_to_folder = '/Users/cadentait/python/github portfolio/Age Detection/Train_/'+c
            shutil.move(get_image, move_to_folder)
            # Check to see if imaage was moved.
            if os.path.exists(move_to_folder+'/'+i):
                print('FILE SUCCESFULLY TRANSFERED!')
            # If image wasn't moved to correct folder, it will break so as not to lose multiple files.
            else:
                print('FILE', get_image, 'MAY BE LOST!!! BREAKING!!!!')
                break
        # If image path isn't correct, it will throw this error. 
        except FileNotFoundError:
            get_image = os.path.join('/Users/cadentait/python/github portfolio/Age Detection/Train/'+i)
            print('The image at file\n---', get_image, '---\n couldn\'t be found and won\'t be appended to...', os.path.join(get_image, '/Users/cadentait/python/github portfolio/Age Detection/Train_/'+c), '\n\n' )
            continue