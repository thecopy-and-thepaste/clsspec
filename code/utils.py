import random
import string
import re
import numpy as np
import random as rnd

from typing import Generator, Iterable


from conabio_ml.datasets.dataset import Dataset
from conabio_ml_text.preprocessing.preprocessing import Tokens

from conabio_ml_text.utils.constraints import TransformRepresentations as TR


from conabio_ml.utils.logger import get_logger, debugger

log = get_logger(__name__)
debug = debugger.debug

re_numbers = re.compile('^[-+]?[\d.]+(?:e-?\d+)?$')


def ood_preproc(item: str):

    tokens = []
    # We only care to remove hyperlink, puntuation, and numbers.
    item = item.lower()

    item = item.translate(str.maketrans(
        string.punctuation, ' '*len(string.punctuation)))
    item = item.replace(" ", "_")

    for token in item.split():
        if re.findall(re_numbers, token):
            continue

        tokens.append(token)

    ix = 0
    while ix < len(tokens):
        step = random.randint(1, 3)

        yield " ".join(tokens[ix: ix+step])
        ix += step


def nchars(preproc_args: dict =
           {
               "pad_size": -1,
               "nchar_size": 2,
               "unk_token": Tokens.UNK_TOKEN
           },
           item: str = ""):
    pad_size = preproc_args.get("pad_size", -1)
    unk_token = preproc_args.get("unk_token", Tokens.UNK_TOKEN)
    nchar_size = preproc_args.get("nchar_size", 2)

    chars = [c for c in item]

    if len(chars) < pad_size:
        pad = [unk_token] * (pad_size - len(chars))
        chars = pad + chars

    nchars = " ".join(["".join(chars[nc: nc + nchar_size])
                       for nc in range(0, len(chars) - nchar_size + 1)])

    return nchars


def datagen(dataset: Dataset.DatasetType,
            X: np.array,
            Y: np.array,
            shuffle: bool = True,
            **kwargs) -> Generator:
    """
    Create the datagen to perform the training, we augment the data if the following manner.

    For every sample, we return
    1 Whole sample
    2 Samples with one token (no PAD) as UNK
    1 Sample with the 2 tokens (no PAD) as UNK

    Returns
    -------
    [type]
        [description]

    Yields
    -------
    [type]
        [description]
    """

    assert len(X) == len(Y), \
        (f"Both X and Y arrays must have to be of the same size")
    _X = X
    _Y = Y

    vocab = {v: k for k, v in enumerate(dataset.representations[TR.VOCAB])}

    batch_size = kwargs["batch_size"]
    sample_length = kwargs["pad_length"]
    unk_token = kwargs["unk_token"]

    label_length = _Y.shape[1]

    unk_ix = vocab[unk_token]

    sample_batch = batch_size // 4
    num_batches = len(_X) // sample_batch

    batch_ixs = list(range(0, len(_X)))

    def data_generator():
        if shuffle:
            rnd.shuffle(batch_ixs)

        batch_ix = 0
        for _ in range(num_batches):

            x_batch = np.full((batch_size, sample_length), unk_ix)
            y_batch = np.full((batch_size, label_length), -1)

            for ix in range(0, batch_size, 4):
                curr_batch = batch_ixs[batch_ix]

                x_batch[ix:ix+4] = _X[curr_batch]
                y_batch[ix:ix+4] = _Y[curr_batch]

                mask = np.where(_X[curr_batch] > 0)[0]
                unk_row_ix = int(np.random.random() * len(mask))
                x_batch[ix + 1][unk_row_ix] = unk_ix
                x_batch[ix + 3][unk_row_ix] = unk_ix

                mask = np.where(_X[curr_batch] > 0)[0]
                unk_row_ix = int(np.random.random() * len(mask))
                x_batch[ix + 2][unk_row_ix] = unk_ix
                x_batch[ix + 3][unk_row_ix] = unk_ix

                batch_ix += 1

            yield (x_batch, y_batch)

    return data_generator
