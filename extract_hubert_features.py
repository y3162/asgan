import os
import argparse
import torch
import torchaudio
import numpy as np
np.float = float
from tqdm import tqdm
from split_data import DIGITS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="path to google speech commands directory",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="path to hubert features directory",
    )
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    hubert = torch.hub.load('RF5/simple-asgan', 'hubert_base')
    device = next(hubert.model.parameters()).device

    total_waves = sum([len(files) if not os.path.basename(cur_dir) in DIGITS else 0 for cur_dir, _, files in os.walk(args.dataset_path)])
    with tqdm(total=total_waves) as bar:
        for cur_dir, _, files in os.walk(args.dataset_path):
            if not os.path.basename(cur_dir) in DIGITS:
                continue

            sub_dir = os.path.relpath(cur_dir, args.dataset_path)
            output_dir = os.path.abspath(os.path.join(args.output_path, sub_dir))
            os.makedirs(output_dir, exist_ok=True)

            for file in files:
                wav, _ = torchaudio.load(os.path.join(cur_dir, file))
                feats = hubert.get_feats_batched(wav.to(device))
                feats = feats.squeeze(0).cpu()
                output_path = os.path.splitext(os.path.join(output_dir, file))[0] + ".pt"
                torch.save(feats, output_path)
                bar.update(1)

if __name__ == "__main__":
    main()
