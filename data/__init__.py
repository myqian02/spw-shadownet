from importlib import import_module
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler

class TrainData():
    def __init__(self, args):
        dataset_name = args.dataset
        
        if hasattr(args, 'train_txt') and args.train_txt:
            from data.txt_dataset import TxtDataset
            dataset = TxtDataset(args)
        else:
            dataset = import_module('data.' + dataset_name.lower())
            dataset = getattr(dataset, dataset_name)(args)
            
        self.loader = DataLoader(
                dataset=dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                drop_last=False,
                )
            
    def get_loader(self):
        return self.loader