import argparse
import glob
import os
import gc

import cv2
import mmcv
import numpy as np
import torch
import torch.nn as nn
from mmengine.runner import load_checkpoint
from mmengine import mkdir_or_exist
from mmcv.image.misc import tensor2imgs
from mmengine import Config
from tqdm import tqdm

#from mmcv.ops.
from updated_builder import build_model

VIDEO_EXTENSIONS = ('.mp4', '.mov', '.mkv')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Inference script of RealBasicVSR')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('input_dir', help='directory of the input video')
    parser.add_argument('output_dir', help='directory of the output video')
    parser.add_argument(
        '--max_seq_len',
        type=int,
        default=15,
        help='maximum sequence length to be processed')
    parser.add_argument(
        '--is_save_as_png',
        type=bool,
        default=True,
        help='whether to save as png')
    parser.add_argument(
        '--fps', type=float, default=25, help='FPS of the output video')
    args = parser.parse_args()

    return args


def init_model(config, checkpoint=None):
    """Initialize a model from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str): Which device the model will deploy. Default: 'cuda:0'.

    Returns:
        nn.Module: The constructed model.
    """

    if isinstance(config, str):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    #config.model.pretrained = None
    #config.test_cfg.metrics = None
    model = build_model(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)

    #model.cfg = config  # save the config in the model for convenience
    model.eval()

    return model

def init_model2(config, checkpoint=None):
    model = None
    return model

def main():
    args = parse_args()

    # initialize the model
    model = init_model(args.config, args.checkpoint)

    # read images
    file_extension = os.path.splitext(args.input_dir)[1]
    
    if file_extension in VIDEO_EXTENSIONS:  # input is a video file
        video_reader = mmcv.VideoReader(args.input_dir)
        inputs = []
        for frame in video_reader:
            inputs.append(np.flip(frame, axis=2))
    elif file_extension == '':  # input is a directory
        inputs = []
        input_paths = sorted(glob.glob(f'{args.input_dir}/*'))
        for input_path in input_paths:
            img = mmcv.imread(input_path, channel_order='rgb')
            inputs.append(img)
    else:
        raise ValueError('"input_dir" can only be a video or a directory.')

    for i, img in enumerate(inputs):
        img = torch.from_numpy(img / 255.).permute(2, 0, 1).float()
        inputs[i] = img.unsqueeze(0)
    inputs = torch.stack(inputs, dim=1)

    # map to cuda, if available
    cuda_flag = False
    if torch.cuda.is_available():
        model = model.cuda()
        cuda_flag = True

    with torch.no_grad():
        if isinstance(args.max_seq_len, int):
            outputs = []
            for i in tqdm(range(0, inputs.size(1), args.max_seq_len)):
                imgs = inputs[:, i:i + args.max_seq_len, :, :, :]
                if cuda_flag:
                    imgs = imgs.cuda()
                #output1 = model(imgs)
                outputs.append(model(imgs).cpu())
            outputs = torch.cat(outputs, dim=1)
        else:
            if cuda_flag:
                #inputs.to(dtype=torch.float16)
                #inputs = inputs.squeeze()
                inputs = inputs.cuda()
            outputs = model(inputs)['output'].cpu()
    del model
    torch.cuda.empty_cache() 
    gc.collect()
    if os.path.splitext(args.output_dir)[1] in VIDEO_EXTENSIONS:
        output_dir = os.path.dirname(args.output_dir)
        mkdir_or_exist(output_dir)

        h, w = outputs.shape[-2:]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.output_dir, fourcc, args.fps,
                                       (w, h))
        for i in tqdm(range(0, outputs.size(1))):
            #img = tensor2imgs(outputs[:, i, :, :, :])[0]
            #the new version of tensor2imgs doesnt multiply the outputs by 255 so it needs 
            #to be done here, i don't think it causes any unwanted side effects
            img = tensor2imgs(outputs[:, i, :, :, :]*255)[0]
            video_writer.write(img)

        cv2.destroyAllWindows()
        video_writer.release()
    else:
        mkdir_or_exist(args.output_dir)
        for i in range(0, outputs.size(1)):
            output = tensor2imgs(outputs[:, i, :, :, :])
            filename = os.path.basename(input_paths[i])
            if args.is_save_as_png:
                file_extension = os.path.splitext(filename)[1]
                filename = filename.replace(file_extension, '.png')
            mmcv.imwrite(output, f'{args.output_dir}/{filename}')


if __name__ == '__main__':
    main()
