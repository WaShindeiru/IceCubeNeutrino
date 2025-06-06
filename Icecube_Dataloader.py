from torch.utils.data import Dataset, DataLoader
import pyarrow.parquet as pq
import torch

def process_tabular(df_selected):
    '''
    Processing input tensor.
    Input Tensor : 3D Tensor
    With time, batch_id, charge.
    '''
    torch_tensor = torch.tensor(df_selected.astype(int).values)
    torch_tensor_tf = torch_tensor.unsqueeze(-1)
    torch_tensor_tf = torch_tensor_tf.type(torch.LongTensor)
    torch_tensor_tf = torch_tensor_tf.permute(1,2,0)
    torch_tensor_tf = torch_tensor_tf.float()
    return(torch_tensor_tf)

class ICECUBE_Dataset(Dataset):
    def __init__(self, parquetfile, batch_dir, batch_num,mode='Train'):
        self.batch_num = batch_num
        self.train_meta = pq.read_table(parquetfile,filters=[('batch_id','==',self.batch_num)]).to_pandas().reset_index(drop = True)
        # Appending only HQS
        self.sensor_info_df = pq.read_table(batch_dir+'batch_'+str(batch_num)+'.parquet',filters=[('auxiliary','==',False)]).to_pandas()
        self.sensor_info_df = self.sensor_info_df.drop('auxiliary',axis=1).reset_index()
        self.dataset_mode = mode
    def __len__(self):
        return self.sensor_info_df.event_id.nunique()

    def __getitem__(self, idx):
        '''
        Tensors will be iterated using event_id, in DataLoader.
        Since Parquet batch files are quite large, parquet files of interest loaded in __init__ function.
        And then generate input tensors using event_id as an index.
        '''
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
    '''
    Resize 3D Tensors to the biggest tensor in a single batch.
    This code would resize all tensors in each batch to the tensor which have largest size (w*h).
    This collate function is applied for batch training, which requires same size of inputs to be feeded.
    '''

    # Find the largest width and height in the batch
    max_width = max(dat['input_tensor'].shape[2] for dat in batch)
    max_height = max(dat['input_tensor'].shape[1] for dat in batch)
    channels = max(dat['input_tensor'].shape[0] for dat in batch)

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