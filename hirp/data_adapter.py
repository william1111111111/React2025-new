"""Field mapping only; no legacy model or dataset imports or normalization changes."""


def generator_inputs(batch):
    return {**{key: batch[key] for key in
               ('speaker_audio', 'speaker_emotion', 'speaker_3dmm')},
            'lengths': batch['lengths'] if 'lengths' in batch else batch['length']}
