import pandas as pd
import numpy as np
import wfdb
import ast
import os
import pickle
from tqdm import tqdm

raw_data_path = 'E:/pv/WORKING/ECG_main_folder/ECG_Classification_MI_detect/data/raw_data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
save_path = 'E:/pv/WORKING/ECG_main_folder/ECG_Classification_MI_detect/data/loaded_data_MI_subclass/'
os.makedirs(save_path, exist_ok=True)
sampling_rate = 100

# Tạo thư mục MI và NORM
ami_path = os.path.join(save_path, 'AMI')
imi_path = os.path.join(save_path, 'IMI')
os.makedirs(ami_path, exist_ok=True)
os.makedirs(imi_path, exist_ok=True)

# Đọc label và map subclass MI 
def aggregate_diagnostic(y_dict, agg_df, weight_threshold=80):
    tmp = []
    for key, value in y_dict.items():
        if value >= weight_threshold and key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_subclass)
    return list(set(tmp))

# Load metadata
Y = pd.read_csv(os.path.join(raw_data_path, 'ptbxl_database.csv'), index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

agg_df = pd.read_csv(os.path.join(raw_data_path, 'scp_statements.csv'), index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

Y['diagnostic_subclass'] = Y.scp_codes.apply(lambda x: aggregate_diagnostic(x, agg_df))

Y = Y[Y['diagnostic_subclass'].apply(lambda x: len(x) > 0)].copy()

# Loại mixed NORM với bệnh
def is_mixed_with_norm(labels):
    return 'NORM' in labels and any(label != 'NORM' for label in labels)

Y = Y[~Y['diagnostic_subclass'].apply(is_mixed_with_norm)].copy()

# Chỉ giữ lại các record chứa MI 
def keep_only_mi(labels):
    if 'IMI' in labels:
        return 'IMI'
    elif 'AMI' in labels:
        return 'AMI'
    else:
        return None

Y['target'] = Y['diagnostic_subclass'].apply(keep_only_mi)
Y = Y[Y['target'].notnull()].copy()

# Cập nhật filepath
if sampling_rate == 100:
    Y['filepath'] = Y['filename_lr']
else:
    Y['filepath'] = Y['filename_hr']

# Tách MI và NORM
Y_ami = Y[Y['target'] == 'AMI']
Y_imi = Y[Y['target'] == 'IMI']

# Lưu record vào đúng folder MI/NORM
count_ami = 0
count_imi = 0

for ecg_id, row in tqdm(Y.iterrows(), total=Y.shape[0], desc="Saving records"):
    filepath = os.path.join(raw_data_path, row['filepath'])
    signal, _ = wfdb.rdsamp(filepath)
    signal = signal.T  # Transpose về (12, 5000)
    signal = signal.astype(np.float32)

    if row['target'] == 'AMI':
        np.save(os.path.join(ami_path, f'record_{ecg_id}.npy'), signal)
        count_ami += 1
    elif row['target'] == 'IMI':
        np.save(os.path.join(imi_path, f'record_{ecg_id}.npy'), signal)
        count_imi += 1

# Lưu DataFrame đã xử lý
with open(os.path.join(save_path, 'Y.pkl'), 'wb') as f:
    pickle.dump(Y, f)

print(f"\n✅ Saved processed data to: {save_path}")
print(f"📊 AMI samples  : {count_ami}")
print(f"📊 IMI samples: {count_imi}")