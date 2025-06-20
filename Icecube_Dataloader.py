from torch.utils.data import Dataset, DataLoader
import pyarrow.parquet as pq
import torch
import pandas as pd
import math
import numpy as np

def convert_to_polar(x, y, z):
    r = math.sqrt(x * x + y * y + z * z)
    c = -1 if y < 0 else 1
    azimuth = math.acos(x / math.sqrt(x * x + y * y)) * c
    zenith = math.acos(z / r)
    return azimuth, zenith


def process_tabular(df_selected):
    return (
        torch.tensor(df_selected.astype(int).values)
        .unsqueeze(-1)
        .long()
        .permute(1, 2, 0)
        .float()
    )

def convert_to_polar_vectorized(df):
    df_new = df.copy()
    r = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
    c = np.where(df['y'] < 0, -1, 1)
    azimuth = np.arccos(df['x'] / np.sqrt(df['x']**2 + df['y']**2)) * c
    zenith = np.arccos(df['z'] / r)
    df_new['azimuth'] = azimuth
    df_new['zenith'] = zenith

    assert (df_new['azimuth'] < 2 * np.pi).all(), "Some azimuth values are >= 2π"
    assert (df_new['zenith'] < np.pi).all(), "Some zenith values are >= π"

    df_new.drop(columns=['x', 'y', 'z'], inplace=True)

    return df_new

class IceCube_Dataloader(Dataset):
    def __init__(self, parquetfile, batch_dir, geometry, batch_num,mode='Train'):
        self.batch_num = batch_num
        self.train_meta = pq.read_table(parquetfile,filters=[('batch_id','==',self.batch_num)]).to_pandas().reset_index(drop = True)
        self.sensor_info_df = pq.read_table(batch_dir+'batch_'+str(batch_num)+'.parquet',filters=[('auxiliary','==',False)]).to_pandas()
        self.sensor_info_df = self.sensor_info_df.drop('auxiliary',axis=1).reset_index()
        self.geometry_info = pd.read_csv(geometry)
        self.dataset_mode = mode

    def __len__(self):
        return self.sensor_info_df.event_id.nunique()

    def __getitem__(self, idx):
        event_id = self.sensor_info_df.event_id.unique()
        sensor_info_df_tmp = self.sensor_info_df
        sensor_info_df_tmp = sensor_info_df_tmp[sensor_info_df_tmp.event_id==event_id[idx]].drop('event_id',axis=1)
        train_meta_tmp = self.train_meta[self.train_meta.event_id==event_id[idx]]
        sensor_info_df_tmp = sensor_info_df_tmp.merge(self.geometry_info, left_on='sensor_id', right_on='sensor_id', how='left')
        sensor_info_df_tmp = convert_to_polar_vectorized(sensor_info_df_tmp)
        input_tensor = process_tabular(sensor_info_df_tmp)
        if self.dataset_mode=='Train':
            label = torch.Tensor(train_meta_tmp[['azimuth','zenith']].values).squeeze()
            sample = {'input_tensor': input_tensor,'label':label}
        else :
            sample = {'input_tensor':input_tensor}
        return sample

def collate_fn(batch):
    max_width = max(dat['input_tensor'].shape[2] for dat in batch)
    max_height = max(dat['input_tensor'].shape[1] for dat in batch)

    resized_batch = []
    for dat in batch:
        tensor = dat['input_tensor']
        resized_tensor = torch.zeros((tensor.shape[0], max_height, max_width), dtype=tensor.dtype)
        resized_tensor[:, :tensor.shape[1], :tensor.shape[2]] = tensor
        resized_batch.append(resized_tensor)

    labels = [tensor['label'] for tensor in batch]

    labels = torch.stack(labels).squeeze(-1)
    resized_batch = torch.stack(resized_batch)

    samples = {'input_tensor' : resized_batch, 'label':labels}

    return samples