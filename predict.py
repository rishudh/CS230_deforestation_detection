import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from unet import UNet

def predict_img(net,
                input_array,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    # Preprocess the numpy array
    input_array = input_array.astype(np.float32)  # Ensure the array is float32
    if scale_factor != 1:
        input_array = torch.nn.functional.interpolate(
            torch.tensor(input_array).unsqueeze(0),
            scale_factor=scale_factor,
            mode='bilinear',
            align_corners=True
        ).squeeze(0).numpy()

    # Add batch and channel dimensions if needed
    
    input_tensor = torch.from_numpy(input_array).unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(input_tensor).cpu()
        output = F.interpolate(output, size=(input_array.shape[1], input_array.shape[2]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input .npy arrays')
    parser.add_argument('--model', '-m', default='/content/Pytorch-UNet/checkpoints/checkpoint_epoch1.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input .npy arrays', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output masks')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1.0,
                        help='Scale factor for the input arrays')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.npy'

    return args.output or list(map(_generate_name, args.input))


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting mask for file {filename} ...')

        # Load .npy file
        input_array = np.load(filename)

        # Predict mask
        mask = predict_img(net=net,
                           input_array=input_array,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_filename = out_files[i]
            np.save(out_filename, mask)  # Save as .npy file
            logging.info(f'Mask saved to {out_filename}')
