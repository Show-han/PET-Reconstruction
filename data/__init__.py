'''create dataset and dataloader'''
import logging
from re import split
import torch.utils.data
import os

def create_dataloader(dataset, dataset_opt, phase):
    '''create dataloader '''
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['num_workers'],
            pin_memory=True)
    elif phase == 'val':
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))


def create_dataset(dataset_opt, phase):
    '''create dataset'''
    mode = dataset_opt['mode']
    from data.LRHR_dataset import LRHRDataset as D
    dataset = D(dataroot=dataset_opt['dataroot'],
                datatype=dataset_opt['datatype'],
                l_resolution=dataset_opt['l_resolution'],
                r_resolution=dataset_opt['r_resolution'],
                split=phase,
                data_len=dataset_opt['data_len'],
                need_LR=(mode == 'LRHR')
                )
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
if __name__ == "__main__":
    from data.LRHR_dataset import LRHRDataset as D
    dataset = D(
        dataroot='C:\\Users\wang\Desktop\PET-Reconstruction-with-Diffusion\dataset\processed',
        datatype='jpg',
        l_resolution=64,
        r_resolution=64,
        split='train',
        data_len=-1,
        need_LR=False
    )
    train_set = dataset
    train_loader=torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle="true",
        num_workers=0,
        pin_memory=True)
    for _, train_data in enumerate(train_loader):
        print(train_data['LP'].shape)
    #     print(torch.zeros(train_data['HR'].shape[0:2], dtype=torch.float))
    # path = 'dataset/processed'
    # # print(os.path.join(path.split(path.split('\\')[-1])[0]),'heihei',path.split('\\')[-1])
    # print(os.path.join(
    #         path.split(path.split('/')[-1])[0],'PreNet', 'I{}_E{}_gen.pth'))