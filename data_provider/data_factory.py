from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour, # 1h
    'ETTh2': Dataset_ETT_hour, # 1h
    'ETTm1': Dataset_ETT_minute, # 15m
    'ETTm2': Dataset_ETT_minute, # 15m
    'weather': Dataset_Custom, # 1h
    'traffic': Dataset_Custom,
    'exchange': Dataset_Custom, # 1d
    'ecl': Dataset_Custom, # 1h
    'pump': Dataset_Custom, # 15m
    'bikerent': Dataset_Custom, # 1h
    'airquality': Dataset_Custom, # 1h
    'stock': Dataset_Custom,
    'copper': Dataset_Custom, # 1d
    'patient': Dataset_Custom,
    'custom': Dataset_Custom,
}


def data_provider(args, flag, max_val):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        reweight=args.reweight,
        max_val=max_val
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
