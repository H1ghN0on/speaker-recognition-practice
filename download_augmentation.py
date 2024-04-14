from sr_labs_book.common import download_dataset, extract_dataset, concatenate, download_protocol, part_extract, split_musan
import os

save_directory = "./data"

os.makedirs(save_directory, exist_ok=True) 
 
# Download SLR17 (MUSAN) and SLR28 (RIR noises) datasets
with open(f'./sr_labs_book/data/lists/augment_datasets.txt', 'r') as f:
    lines = f.readlines()
    
download_dataset(lines, user=None, password=None, save_path=f'{save_directory}')


# Extract SLR17 (MUSAN)
extract_dataset(save_path=f'{save_directory}', fname=f'{save_directory}/musan.tar.gz')

# Extract SLR28 (RIR noises)
part_extract(save_path=f'{save_directory}', fname=f'{save_directory}/rirs_noises.zip', target=['RIRS_NOISES/simulated_rirs/mediumroom', 'RIRS_NOISES/simulated_rirs/smallroom'])

# Split MUSAN (SLR17) dataset for faster random access
split_musan(save_path=f'{save_directory}')
