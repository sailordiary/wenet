import argparse
from functools import partial
import sys

from langid.langid import json
import torch
from wenet.dataset import processor
from wenet.experimental.dmels.processor import compute_melspectrogram

from wenet.dataset.datapipes import TextLineDataPipe


def parse_json(elem):
    line = elem['line']
    obj = json.loads(line)
    obj['file_name'] = elem['file_name']
    return dict(obj)


def get_dataset(data_list, mel_conf: dict):
    dataset = TextLineDataPipe(data_list)

    dataset = dataset.map(parse_json)
    dataset = dataset.map(processor.decode_wav)
    # TODO: move sample_rate to args
    dataset = dataset.map(
        partial(processor.resample, resample_rate=mel_conf['sample_rate']))

    # TODO: move melspectrogram conf to config
    dataset = dataset.map(partial(compute_melspectrogram, **mel_conf))

    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='compute dmel\'s m and M')

    parser.add_argument('--data_list', type=str, required=True)
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--hop_length', type=int, default=160)
    parser.add_argument('--n_fft', type=int, default=400)
    parser.add_argument('--n_mels', type=int, default=128)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument("--log_interval", type=int, default=1000)
    args = parser.parse_args()

    mel_conf = {
        "sample_rate": args.sample_rate,
        "hop_length": args.hop_length,
        "n_fft": args.n_fft,
        "n_mels": args.n_mels,
    }
    dataset = get_dataset(args.data_list, mel_conf)
    m = torch.tensor(-1, dtype=torch.float32)
    M = torch.tensor(0, dtype=torch.float32)
    for (i, d) in enumerate(dataset):
        mel_spec = d['feat']
        m_ = torch.min(mel_spec)
        M_ = torch.max(mel_spec)

        m = torch.where(m_ < m, m_, m)
        M = torch.where(M_ > M, M_, M)

        if i % args.log_interval == 0:
            print(f'processed {i} wavs, curtent m: {m}, M: {M}',
                  file=sys.stderr,
                  flush=True)

    with open(args.output_file, 'w') as f:
        f.write(json.dumps({
            m: m.numpy(),
            M: M.numpy(),
        }) + '\n')
