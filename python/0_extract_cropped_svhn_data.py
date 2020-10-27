import zipfile

# You can replace your classes of dataset to data folder.
# It is clear that you don't need to run this script when you want to train your own data
with zipfile.ZipFile('data/cropped_svhn_data.zip', 'r') as zip_ref:
    zip_ref.extractall('data')
