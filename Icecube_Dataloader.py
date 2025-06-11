from torch.utils.data import Dataset, DataLoader
import pyarrow.parquet as pq
import torch

def process_tabular(df_selected):
    return (
        torch.tensor(df_selected.astype(int).values)
        .unsqueeze(-1)
        .long()
        .permute(1, 2, 0)
        .float()
    )

class IceCube_Dataloader(Dataset):
    def __init__(self, parquetfile, batch_dir, batch_num,mode='Train'):
        self.batch_num = batch_num
        self.train_meta = pq.read_table(parquetfile,filters=[('batch_id','==',self.batch_num)]).to_pandas().reset_index(drop = True)
        self.sensor_info_df = pq.read_table(batch_dir+'batch_'+str(batch_num)+'.parquet',filters=[('auxiliary','==',False)]).to_pandas()
        self.sensor_info_df = self.sensor_info_df.drop('auxiliary',axis=1).reset_index()
        self.dataset_mode = mode

    def __len__(self):
        return self.sensor_info_df.event_id.nunique()

    def __getitem__(self, idx):
        event_id = self.sensor_info_df.event_id.unique()
        sensor_info_df_tmp = self.sensor_info_df
        sensor_info_df_tmp = sensor_info_df_tmp[sensor_info_df_tmp.event_id==event_id[idx]].drop('event_id',axis=1)
        train_meta_tmp = self.train_meta[self.train_meta.event_id==event_id[idx]]
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