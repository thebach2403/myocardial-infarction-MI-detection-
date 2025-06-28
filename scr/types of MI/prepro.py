import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
import scipy.signal as sg
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from scipy.signal import butter, filtfilt

##################### Global variables #################################
loaded_path = 'E:/pv/WORKING/ECG_main_folder/ECG_Classification_MI_detect/data/loaded_data_MI_subclass/'
save_path = 'E:/pv/WORKING/ECG_main_folder/ECG_Classification_MI_detect/data/preprocessed_MI_subclass/'
os.makedirs(save_path, exist_ok=True)

ami_path = os.path.join(loaded_path, 'AMI')
imi_path = os.path.join(loaded_path, 'IMI')

sampling_rate = 100
num_leads = 12
desired_length = 4096  # lấy 4096 samples đầu tiên cho mỗi lead

##################### Các hàm preprocess ################################

def butter_bandpass_filter(signal, lowcut=1.0, highcut=45.0, fs=sampling_rate, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    
    padlen = 3 * max(len(a), len(b))
    if len(signal) <= padlen:
        raise ValueError(f"Signal length {len(signal)} too short for filtfilt (requires > {padlen} samples).")
    
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


def z_score_norm(signal):
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

def process_one_record(id_label_split):
    ecg_id, label, split = id_label_split 
    record_path = os.path.join(loaded_path, label, f'record_{ecg_id}.npy')
    record = np.load(record_path)  # shape: (12, length)

    if record.shape[0] != 12:
        print(f"❌ ERROR: Record {ecg_id} has {record.shape[0]} leads instead of 12.")

    processed_record = []

    for lead in range(num_leads):
        raw_signal = record[lead, :]
        
        if len(raw_signal) < 50:
            print(f"⚠️ WARNING: Record {ecg_id} lead {lead} too short: length = {len(raw_signal)}")

        try:
            filtered = butter_bandpass_filter(raw_signal)
        except Exception as e:
            print(f"❌ ERROR filtering record {ecg_id}, lead {lead}: {e}")
            return None

        normalized = z_score_norm(filtered)

        if len(normalized) >= desired_length:
            normalized = normalized[:desired_length]
        else:
            pad_width = desired_length - len(normalized)
            normalized = np.pad(normalized, (0, pad_width), 'constant')

        processed_record.append(normalized)

    processed_record = np.stack(processed_record)

    save_dir = os.path.join(save_path, split, label)
    os.makedirs(save_dir, exist_ok=True)
    save_path_full = os.path.join(save_dir, f'record_{ecg_id}.npy')
    np.save(save_path_full, processed_record)

    return None

#################### MAIN ####################

if __name__ == '__main__':
    for split in ['train', 'val', 'test']:
        for label in ['AMI', 'IMI']:
            os.makedirs(os.path.join(save_path, split, label), exist_ok=True)

    print("Lấy danh sách record...")
    ami_files = [f for f in os.listdir(ami_path) if f.endswith('.npy')]
    imi_files = [f for f in os.listdir(imi_path) if f.endswith('.npy')]

    ami_ids = [int(f.split('_')[1].split('.npy')[0]) for f in ami_files]
    imi_ids = [int(f.split('_')[1].split('.npy')[0]) for f in imi_files]

    ami_labels = ['AMI'] * len(ami_ids)
    imi_labels = ['IMI'] * len(imi_ids)

    all_ids = ami_ids + imi_ids
    all_labels = ami_labels + imi_labels

    # Chia train/val/test
    train_ids, test_ids, train_labels, test_labels = train_test_split(
        all_ids, all_labels, test_size=0.2, stratify=all_labels, random_state=42)
    train_ids, val_ids, train_labels, val_labels = train_test_split(
        train_ids, train_labels, test_size=0.1, stratify=train_labels, random_state=42)

    # Lưu labels
    with open(os.path.join(save_path, 'labels_train.pkl'), 'wb') as f:
        pickle.dump(list(zip(train_ids, train_labels)), f)
    with open(os.path.join(save_path, 'labels_val.pkl'), 'wb') as f:
        pickle.dump(list(zip(val_ids, val_labels)), f)
    with open(os.path.join(save_path, 'labels_test.pkl'), 'wb') as f:
        pickle.dump(list(zip(test_ids, test_labels)), f)

    # Chuẩn bị cho song song
    train_input = [(ecg_id, label, 'train') for ecg_id, label in zip(train_ids, train_labels)]
    val_input   = [(ecg_id, label, 'val')   for ecg_id, label in zip(val_ids, val_labels)]
    test_input  = [(ecg_id, label, 'test')  for ecg_id, label in zip(test_ids, test_labels)]

    print("Processing test data...")
    with ProcessPoolExecutor(max_workers=4) as executor:
        list(tqdm(executor.map(process_one_record, test_input), total=len(test_input)))

    print("Processing validation data...")
    with ProcessPoolExecutor(max_workers=4) as executor:
        list(tqdm(executor.map(process_one_record, val_input), total=len(val_input)))

    print("Processing train data...")
    with ProcessPoolExecutor(max_workers=4) as executor:
        list(tqdm(executor.map(process_one_record, train_input), total=len(train_input)))

    print('✅ Preprocessing complete!')
