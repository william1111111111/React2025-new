"""Timing and quote identity checks with known synthetic CTC paths."""
import pytest
import torch
from semantic_supervision.align.automatic import align,quote_words

def test_repeated_letters_require_blank_and_preserve_two_occurrences():
    labels=('-', '|','A','B')
    path=[0,2,0,2,0,1,0,3,0]
    logits=torch.full((len(path),len(labels)),-12.)
    for i,t in enumerate(path):logits[i,t]=12.
    words=align(logits.log_softmax(-1),'aa b',labels)
    assert [w['word'] for w in words]==['AA','B']
    assert words[0]['start_s']==pytest.approx(.02)
    assert words[0]['end_s']==pytest.approx(.085)
    assert words[1]['start_s']==pytest.approx(.14)
    assert words[0]['end_s']<words[1]['start_s']

def test_repeated_quotes_map_to_distinct_occurrences():
    text='Yes. Then yes. Yes.'
    pos,ids=quote_words(text,'Yes.')
    later,later_ids=quote_words(text,'Yes.',pos)
    assert pos==0 and later==15 and ids==[0] and later_ids==[3]
    with pytest.raises(ValueError):quote_words(text,'Yes.',later)

def test_quote_rewrite_rejected():
    with pytest.raises(ValueError):quote_words("I'm fine.",'I am fine.')
