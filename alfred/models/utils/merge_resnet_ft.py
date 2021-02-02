import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))

import torch
import os
from PIL import Image
from nn.resnet import Resnet
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # settings
    parser.add_argument('--data', help='data folder', default='data/2.1.0')
    parser.add_argument('--filename', help='filename of feat', default='feat_conv.pt')
    parser.add_argument('--img_folder', help='folder containing raw images', default='raw_images_1')
    merge_feat_list = ['feat_conv.pt', 'feat_conv_1.pt', 'feat_conv_2.pt',
                       'feat_exploration_conv.pt', 'feat_exploration_conv_1.pt', 'feat_exploration_conv_2.pt']

    # parser
    args = parser.parse_args()

    # load resnet model
    extractor = Resnet(args, eval=True)
    skipped = []

    for root, dirs, files in os.walk(args.data):
        if os.path.basename(root) == args.img_folder:
            import pdb; pdb.set_trac()
            root = root.rsplit("/", 1)[0]
            merge_tensor_with_dict = {}
            for feat_name in merge_feat_list:
                feat_path = os.path.join(root, feat_name)
                feat_tensor = torch.load(feat_path)
                feat_name = feat_name[:-3]
                merge_tensor_with_dict[feat_name] = feat_tensor
            torch.save(merge_tensor_with_dict, os.path.join(root, "feat_merge.pt"))

    print("Skipped:")
    print(skipped)