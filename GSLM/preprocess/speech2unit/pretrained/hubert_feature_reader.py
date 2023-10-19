# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import fairseq
import librosa
import soundfile as sf
import torch.nn.functional as F


class HubertFeatureReader:
    """
    Wrapper class to run inference on HuBERT model.
    Helps extract features for a given audio file.
    """

    def __init__(self, checkpoint_path, layer, max_chunk=1600000):
        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [checkpoint_path]
        )
        self.model = model[0].eval().cuda()
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk

    def read_audio(self, path, offset=None, duration=None, ref_len=None):
        """
        Read an audio file and return its waveform.

        Args:
            path (str): Path to the audio file.
            offset (float): Starting time offset for reading audio.
            duration (float): Duration of audio to read.
            ref_len (int): Reference length for the audio.

        Returns:
            numpy.ndarray: Audio waveform as a NumPy array.
        """

        # Important! Force load mono and at 16 kHz
        assert self.task.cfg.sample_rate == 16000
        if offset is None and duration is None:
            wav, sr = librosa.load(path, mono=True, sr=self.task.cfg.sample_rate)
        else:
            # Handle 'None' as a string and convert it to actual None
            duration = None if duration == "None" else float(duration)
            wav, sr = librosa.load(path, offset=offset, duration=duration, mono=True, sr=self.task.cfg.sample_rate)

        assert sr == self.task.cfg.sample_rate == 16000, sr

        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            print(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def get_feats(self, file_path, offset=None, duration=None, ref_len=None):
        x = self.read_audio(file_path, offset, duration, ref_len)
        with torch.no_grad():
            x = torch.from_numpy(x).float().cuda()
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start: start + self.max_chunk]
                feat_chunk, _ = self.model.extract_features(
                    source=x_chunk,
                    padding_mask=None,
                    mask=False,
                    output_layer=self.layer,
                )
                feat.append(feat_chunk)
        return torch.cat(feat, 1).squeeze(0)
