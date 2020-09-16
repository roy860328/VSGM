import os
import torch
vocab = torch.load("../data/json_feat_2.1.0/pp.vocab")
action_low_word = vocab['action_low'].index2word(list(range(0, len(vocab['action_low']))))
action_high_word = vocab['action_high'].index2word(list(range(0, len(vocab['action_high']))))
word_word = vocab['word'].index2word(list(range(0, len(vocab['word']))))
print(action_low_word)
print(action_high_word)
print(word_word)