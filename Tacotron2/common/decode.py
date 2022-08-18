from kaldi import nnet3
from kaldi.util.io import xopen
from kaldi import hmm
from kaldi import fstext


def read_nnet3_model(model_path: str) -> nnet3.Nnet:
    """Read in a nnet3 spk_embedder in raw format.

    Actually if this spk_embedder is not a raw format it will still work, but this is
    not an official feature; it was due to some kaldi internal code.

    Args:
        model_path: Path to a raw nnet3 spk_embedder, e.g., "data/final.raw"

    Returns:
        nnet: A neural network AM.
    """
    nnet = nnet3.Nnet()
    with xopen(model_path) as istream:
        nnet.read(istream.stream(), istream.binary)
    return nnet


def read_trans_model(model_path: str) -> hmm.TransitionModel:
    """Read in a transition spk_embedder stored in the header of a .mdl file.

    Args:
        model_path: Path to a .mdl file.

    Returns:
        The transition spk_embedder.
    """
    with xopen(model_path) as istream:
        trans_model = hmm.TransitionModel().read(istream.stream(),
                                                 istream.binary)
    return trans_model


def read_den_fst(fst_path: str) -> fstext.StdVectorFst:
    """Read in a dense FST file.

    Args:
        fst_path: Path to a .fst file.

    Returns:
        The dense FST.
    """
    den_fst = fstext.StdVectorFst.read(fst_path)
    return den_fst
