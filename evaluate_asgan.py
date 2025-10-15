import os
import argparse
from omegaconf import OmegaConf
import torch
import torchaudio

from hubconf import hubert_hifigan, Conv2SC09Wrapper
from density.config import TrainConfig
from density.models import RP_W

def generate_test_samples(model, sample_root_dir, n=100, N=5000):
    assert N % n == 0
    if os.path.exists(sample_root_dir):
        return
    os.makedirs(sample_root_dir)
    
    model.eval()
    with torch.no_grad():
        for i in range(0, N, n):
            audio = model.unconditional_generate(n)
            audio = audio.cpu()
            for j in range(n):
                torchaudio.save(
                    filepath=os.path.join(sample_root_dir, f"{i+j:05d}.wav"),
                    src=audio[j].unsqueeze(0),
                    sample_rate=model.seq_len,
                )

def main():
    parser = argparse.ArgumentParser(usage="\n" + "-"*10 + " Default config " + "-"*10 + "\n" + str(OmegaConf.to_yaml(OmegaConf.structured(TrainConfig))))
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="path to RP_W checkpoint",
    )
    parser.add_argument(
        "--sample-root-path",
        type=str,
        required=True,
        help="path to directory for sample wav files",
    )
    args, unknown_args = parser.parse_known_args()
    override_cfg = OmegaConf.from_cli(unknown_args)
    base_cfg = OmegaConf.structured(TrainConfig)
    cfg: TrainConfig = OmegaConf.merge(base_cfg, override_cfg)

    device = "cuda:0"

    generator = RP_W(cfg.rp_w_cfg)
    generator.load_state_dict(torch.load(args.checkpoint_path)["g_state_dict"])
    generator.to(device)

    hifigan = hubert_hifigan()
    hifigan.to(device)

    model = Conv2SC09Wrapper(generator, hifigan, cfg)

    generate_test_samples(model, args.sample_root_path)

if __name__ == "__main__":
    main()
