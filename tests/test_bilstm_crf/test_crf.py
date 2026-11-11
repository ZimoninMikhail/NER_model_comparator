"""Численные проверки CRF независимым полным перебором коротких цепочек."""

import torch

from src.models.bilstm_crf_core import LinearChainCRF


def test_viterbi_uses_transitions_instead_of_independent_argmax():
    crf = LinearChainCRF(2)
    with torch.no_grad():
        for p in crf.parameters():p.zero_()
        crf.transitions[1, 1] = 5
    emissions = torch.tensor([[[4.,3.],[4.,3.]]])
    assert emissions.argmax(-1).tolist() == [[0,0]]
    assert crf.decode(emissions).tolist() == [[1,1]]


