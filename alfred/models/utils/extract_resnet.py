import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))

import torch
import os
from PIL import Image
from nn.resnet import Resnet
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import glob
import random

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # settings
    parser.add_argument('--data', help='data folder', default='data/2.1.0')
    parser.add_argument('--batch', help='batch size', default=256, type=int)
    parser.add_argument('--gpu', help='use gpu', action='store_true')
    parser.add_argument('--skip_existing', help='skip folders that already have feats', action='store_true')
    parser.add_argument('--visual_model', default='resnet18', help='model type: maskrcnn or resnet18', choices=['maskrcnn', 'resnet18'])
    parser.add_argument('--keyname', help='keyname of feat', default='feat_conv')
    parser.add_argument('--str_save_ft_name', help='str_save_ft_name', default='')
    parser.add_argument('--img_folder', help='folder containing raw images', default='raw_images')

    # parser
    args = parser.parse_args()

    if args.str_save_ft_name == "":
        raise "str_save_ft_name must have xxxx.pt"
    print(args.str_save_ft_name)
    # load resnet model
    if args.gpu:
        device = "cuda"
    else:
        device = "cpu"
    extractor = Resnet(args, device, eval=True)
    skipped = []

    main_search_img_folder = args.img_folder.split(',')[0]
    roots = []
    for search in ['/train', '/valid_seen', '/valid_unseen']:
        walk = os.walk(args.data + search)
        roots += [root for root, dirs, files in walk if os.path.basename(root) == main_search_img_folder]
    random.shuffle(roots)
    print(len(roots))
    for i, root in enumerate(roots):
        if os.path.basename(root) == main_search_img_folder:
            feat_img_dict = {}
            root = root.replace(main_search_img_folder, '')
            for img_folder, keyname in zip(args.img_folder.split(','), args.keyname.split(',')):
                root_img = os.path.join(root, img_folder)
                files = glob.glob(root_img + '/*.png') +\
                    glob.glob(root_img + '/*.jpg')
                fimages = sorted([f for f in files
                                  if (f.endswith('.png') or (f.endswith('.jpg')))])
                if len(fimages) > 0:
                    if args.skip_existing\
                    and os.path.isfile(os.path.join(root, args.str_save_ft_name))\
                    and os.stat(os.path.join(root, args.str_save_ft_name)).st_size>100:
                        print("skip_existing")
                        break
                    try:
                        print('{}'.format(root_img))
                        image_loader = Image.open if isinstance(fimages[0], str) else Image.fromarray
                        images = [image_loader(f).convert('RGB') for f in fimages]
                        feat = extractor.featurize(images, batch=args.batch)
                        feat_img_dict[keyname] = feat.cpu()
                    except Exception as e:
                        print(e)
                        print("Skipping " + root_img)
                        skipped.append(root_img)
                        break

                else:
                    print('empty; skipping {}'.format(root_img))
            if len(feat_img_dict):
                torch.save(feat_img_dict, os.path.join(root, args.str_save_ft_name))
                print("i: ", i)

    print("Skipped:")
    print(skipped)