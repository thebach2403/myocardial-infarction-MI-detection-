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

# Các subclass MI cần phân loại
mi_subclasses = {
    'IMI': 'Inferior MI',
    'AMI': 'Anterior MI',
    'LMI': 'Lateral MI',
    'PMI': 'Posterior MI'
}

# Tạo thư mục cho mỗi subclass MI và NORM
for subclass in mi_subclasses.keys():
    os.makedirs(os.path.join(save_path, subclass), exist_ok=True)
os.makedirs(os.path.join(save_path, 'NORM'), exist_ok=True)


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

# Remove rows without label
Y = Y[Y['diagnostic_subclass'].apply(lambda x: len(x) > 0)].copy()

# Loại mixed NORM với bệnh
def is_mixed_with_norm(labels):
    return 'NORM' in labels and any(label != 'NORM' for label in labels)

Y = Y[~Y['diagnostic_superclass'].apply(is_mixed_with_norm)].copy()

# Chỉ giữ lại các record chứa MI hoặc NORM
def keep_only_mi_norm(labels):
    if 'MI' in labels:
        return 'MI'
    elif 'NORM' in labels:
        return 'NORM'
    else:
        return None

Y['target'] = Y['diagnostic_superclass'].apply(keep_only_mi_norm)
Y = Y[Y['target'].notnull()].copy()

# Cập nhật filepath
if sampling_rate == 100:
    Y['filepath'] = Y['filename_lr']
else:
    Y['filepath'] = Y['filename_hr']

from sklearn.utils import resample

# Tách MI và NORM
Y_mi = Y[Y['target'] == 'MI']
Y_norm = Y[Y['target'] == 'NORM']

# Undersample NORM
Y_norm_downsampled = resample(Y_norm,
                              replace=False,    # không lấy lại mẫu
                              n_samples=len(Y_mi)+1000, # cho bằng số lượng MI
                              random_state=42)  # random state để tái lặp được

# Gộp lại
Y_balanced = pd.concat([Y_mi, Y_norm_downsampled])

# Shuffle (trộn lẫn cho đều)
# Y_balanced = Y_balanced.sample(frac=1, random_state=42)

# Lưu record vào đúng folder MI/NORM
count_mi = 0
count_norm = 0

for ecg_id, row in tqdm(Y_balanced.iterrows(), total=Y_balanced.shape[0], desc="Saving records"):
    filepath = os.path.join(raw_data_path, row['filepath'])
    signal, _ = wfdb.rdsamp(filepath)
    signal = signal.T  # Transpose về (12, 5000)
    signal = signal.astype(np.float32)

    if row['target'] == 'MI':
        np.save(os.path.join(mi_path, f'record_{ecg_id}.npy'), signal)
        count_mi += 1
    elif row['target'] == 'NORM':
        np.save(os.path.join(norm_path, f'record_{ecg_id}.npy'), signal)
        count_norm += 1

# Lưu DataFrame đã xử lý
with open(os.path.join(save_path, 'Y.pkl'), 'wb') as f:
    pickle.dump(Y, f)

print(f"\n✅ Saved processed data to: {save_path}")
print(f"📊 MI samples  : {count_mi}")
print(f"📊 NORM samples: {count_norm}")
