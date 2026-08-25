import pandas as pd
from tqdm import tqdm

input_filepath = 'artifacts/fineweb/raw/000_00001.parquet'
df = pd.read_parquet(input_filepath, columns=['text', 'language'])
df = df[df['language'] == 'en']
print(df.head())

save_folder = 'dataset/fineweb'


DATA_PER_FILE = 1024 * 1024 * 64
data_buffer = ""
file_count = 0
for i, row in tqdm(df.iterrows()):
    text = row['text'].encode('ascii', errors="ignore").decode('ascii')
    data_buffer += text
    data_buffer += "\n\n\n\n"
    if len(data_buffer) > DATA_PER_FILE:
        with open(f"{save_folder}/corpus_{file_count}.txt", 'w') as f:
            f.write(data_buffer)
        file_count += 1
        data_buffer = ""

