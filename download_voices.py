from sr_labs_book.common import download_dataset, extract_dataset, concatenate, download_protocol
import os

save_directory = "./data"

os.makedirs(save_directory, exist_ok=True) 
 
with open(f'./sr_labs_book/data/lists/datasets.txt', 'r') as f:
    lines = f.readlines()

download_dataset(lines, user='voxceleb1902', password = 'nx0bl2v2', save_path = save_directory)

# Concatenate archives for VoxCeleb1 dev set
with open('./sr_labs_book/data/lists/concat_arch.txt', 'r') as f:
    lines = f.readlines()

concatenate(lines, save_path=f'{save_directory}')

extract_dataset(save_path=f'{save_directory}/voxceleb1_test', fname=f'{save_directory}/vox1_test_wav.zip')
extract_dataset(save_path=f'{save_directory}/voxceleb1_dev', fname=f'{save_directory}/vox1_dev_wav.zip')

# Download VoxCeleb1 identification protocol
with open('./sr_labs_book/data/lists/protocols.txt', 'r') as f:
    lines = f.readlines()

download_protocol(lines, save_path=f'{save_directory}/voxceleb1_test')