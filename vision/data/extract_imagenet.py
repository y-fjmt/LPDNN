import os
import glob
import argparse
import tarfile
import scipy
from tqdm import tqdm

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--valid', action='store_true')
    args = parser.parse_args()
    
    assert args.train or args.valid, \
        'Either --train or --valid must be specified'
        
    
    # train data
    if args.train:
        target_dir = './ILSVRC2012_img_train/'
        for tar_filepath in tqdm(glob.glob(os.path.join(target_dir, '*.tar'))):
            target_dir = tar_filepath.replace('.tar', '')
            os.mkdir(target_dir)
            with tarfile.open(tar_filepath, 'r') as tar:
                tar.extractall(path=target_dir)
            os.remove(tar_filepath)
        
    
    # validation data
    if args.valid:
        
        imagenet_valid_tar_path = './ILSVRC2012_img_val.tar'
        target_dir = './ILSVRC2012_img_val_for_ImageFolder'
        meta_path = './ILSVRC2012_devkit_t12/data/meta.mat'
        trueth_label_path = './ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'
        
        meta = scipy.io.loadmat(meta_path, squeeze_me=True)
        ilsvrc2012_id_to_wnid = {m[0]: m[1] for m in meta['synsets']}

        with open(trueth_label_path, 'r') as f:
            ilsvrc_ids = tuple(int(ilsvrc_id) for ilsvrc_id in f.read().split('\n')[:-1])

        for ilsvrc_id in ilsvrc_ids:
            wnid = ilsvrc2012_id_to_wnid[ilsvrc_id]
            os.makedirs(os.path.join(target_dir, wnid), exist_ok=True)

        os.makedirs(target_dir, exist_ok=True)
        
        num_valid_images = 50000
        with tarfile.open(imagenet_valid_tar_path, mode='r') as tar:
            for valid_id, ilsvrc_id in zip(range(1, num_valid_images+1), ilsvrc_ids):
                wnid = ilsvrc2012_id_to_wnid[ilsvrc_id]
                filename = 'ILSVRC2012_val_{}.JPEG'.format(str(valid_id).zfill(8))
                print(filename, wnid)
                img = tar.extractfile(filename)
                with open(os.path.join(target_dir, wnid, filename), 'wb') as f:
                    f.write(img.read())
    