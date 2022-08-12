from argparse import ArgumentParser

import torch
from soundfile import write
from torchaudio import load

from sgmse.model import ScoreModel
from sgmse.util.other import pad_spec

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--filename", type=str, required=True, help='A file to enhance')
    parser.add_argument("--enhanced_filename", type=str, required=True, help='An enhanced filename')
    parser.add_argument("--ckpt", type=str, required=True,   help='Path to model checkpoint.')
    parser.add_argument("--corrector", type=str, choices=("ald", "langevin", "none"), default="ald", help="Corrector class for the PC sampler.")
    parser.add_argument("--corrector_steps", type=int, default=1, help="Number of corrector steps")
    parser.add_argument("--snr", type=float, default=0.5, help="SNR value for (annealed) Langevin dynmaics.")
    parser.add_argument("--N", type=int, default=30, help="Number of reverse steps")
    args = parser.parse_args()

    filename = args.filename
    enhanced_filename = args.enhanced_filename
    checkpoint_file = args.ckpt
    corrector_cls = args.corrector

    # Settings
    sr = 16000
    snr = args.snr
    N = args.N
    corrector_steps = args.corrector_steps

    # Load score model 
    model = ScoreModel.load_from_checkpoint(checkpoint_file, base_dir='', batch_size=16, num_workers=0, kwargs=dict(gpu=False))
    model.eval(no_ema=False)
    model.cuda()

    # Load wav
    y, _ = load(filename) 
    T_orig = y.size(1)   

    # Normalize
    norm_factor = y.abs().max()
    y = y / norm_factor
        
    # Prepare DNN input
    Y = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
    Y = pad_spec(Y)
        
    # Reverse sampling
    sampler = model.get_pc_sampler(
        'reverse_diffusion', corrector_cls, Y.cuda(), N=N, 
        corrector_steps=corrector_steps, snr=snr)
    sample, _ = sampler()
        
    # Backward transform in time domain
    x_hat = model.to_audio(sample.squeeze(), T_orig)

    # Renormalize
    x_hat = x_hat * norm_factor

    # Write enhanced wav file
    write(enhanced_filename, x_hat.cpu().numpy(), 16000)
