""" Sanity Checks for project

Authors:

"""

import json
import math
import pickle
import sys
import time

import numpy as np

from docopt import docopt
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from utils import pad_sents_char, read_corpus, batch_iter
from vocab import Vocab, VocabEntry

from char_decoder import CharDecoder
from nmt_model import NMT

from highway import Highway


import torch
import torch.nn as nn
import torch.nn.utils

#----------
# CONSTANTS
#----------
BATCH_SIZE = 5
EMBED_SIZE = 3
HIDDEN_SIZE = 3
DROPOUT_RATE = 0.0


# Sanity check for Embedding layer

def sanity_check_embedding_layer(model):
	""" Sanity check for model_embeddings.py
		basic shape check
	"""
	print ("-"*80)
	print("Running Sanity Check for Model Embedding")
	print ("-"*80)
	sentence_length = 10
	max_word_length = 21
	inpt = torch.zeros(sentence_length, BATCH_SIZE, max_word_length, dtype=torch.long)
	ME_source = model.model_embeddings_source
	output = ME_source.forward(inpt)
	output_expected_size = [sentence_length, BATCH_SIZE, EMBED_SIZE]
	assert(list(output.size()) == output_expected_size), "output shape is incorrect: it should be:\n {} but is:\n{}".format(output_expected_size, list(output.size()))
	print("Sanity Check Passed for Question 1j: Model Embedding!")
	print("-"*80)

# Sanity check for Highway

# Sanity check for Encoder

# Sanity check for BIDAF outputs




###
def main():
    """ Main func.
    """
    args = docopt(__doc__)

    # Check Python & PyTorch Versions
    assert (sys.version_info >= (3, 5)), "Please update your installation of Python to version >= 3.5"
    assert(torch.__version__ == "1.0.0"), "Please update your installation of PyTorch. You have {} and you should have version 1.0.0".format(torch.__version__)

    # Seed the Random Number Generators
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    vocab = Vocab.load('./sanity_check_en_es_data/vocab_sanity_check.json')

    # Create NMT Model
    model = NMT(
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        dropout_rate=DROPOUT_RATE,
        vocab=vocab)

    char_vocab = DummyVocab()

    # Initialize CharDecoder
    decoder = CharDecoder(
        hidden_size=HIDDEN_SIZE,
        char_embedding_size=EMBED_SIZE,
        target_vocab=char_vocab)

    if args['1e']:
        question_1e_sanity_check()
    elif args['1f']:
        question_1f_sanity_check()
    elif args['1j']:
        question_1j_sanity_check(model)
    elif args['2a']:
        question_2a_sanity_check(decoder, char_vocab)
    elif args['2b']:
        question_2b_sanity_check(decoder, char_vocab)
    elif args['2c']:
        question_2c_sanity_check(decoder)
    elif args['2d']:
        question_2d_sanity_check(decoder)
    elif args['1h']:
        question_1h_sanity_check()
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()

