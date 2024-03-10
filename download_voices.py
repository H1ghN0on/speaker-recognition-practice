from sr_labs_book.common import download_dataset, extract_dataset
import os

os.makedirs("./data", exist_ok=True) 
save_directory = "./data"

with open(f'./sr_labs_book/data/lists/datasets.txt', 'r') as f:
    lines = f.readlines()

download_dataset(lines, user='voxceleb1902', password = 'nx0bl2v2', save_path = save_directory)
extract_dataset(save_path=f'{save_directory}/voxceleb1_test', fname=f'{save_directory}/vox1_test_wav.zip')