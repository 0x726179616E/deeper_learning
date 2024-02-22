#!/usr/bin/env python3

import os
import torch
import torch.nn as nn

# detect torch device
if torch.cuda.is_available(): device = 'cuda' 
elif torch.backends.mps.is_available(): device = 'mps'
else: device = 'cpu'

# TODO: implement long-short term memory (LSTM) recurrent neural network model

# all data preparation is from: https://github.com/karpathy/makemore
class CharDataset(torch.utils.data.Dataset):
    def __init__(self, words, chars, max_word_length):
        self.words = words
        self.chars = chars 
        self.max_word_length = max_word_length
        self.stoi = {ch:i+1 for i,ch in enumerate(chars)} # string to integer
        self.itos = {i:s for s,i in self.stoi.items()} # integer to string

    def __len__(self):
        return len(self.words)

    def contains(self, word):
        return word in self.words

    def get_vocab_size(self):
        return len(self.chars) + 1 # + 1 for special token
    
    def encode(self, word):
        return torch.tensor([self.stoi[w] for w in word], dtype=torch.long)

    def decode(self, ix):
        return ''.join(self.itos[i] for i in ix)

    def __getitem__(self, idx):
        word = self.words[idx] # retrieve word using provided index
        ix = self.encode(word) # encode word into sequence of integers
        x = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        x[1:1+len(ix)] = ix # input tensor
        y[:len(ix)] = ix # target tensor
        y[len(ix)+1:] = -1 # -1 index masks the loss at the inactive locations
        return x, y

def load(file_path):
    abs_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
    try: 
        with open(abs_file_path, 'r') as f:
            data = f.read()
    except IOError: 
        print(f'error reading file: {file_path}')
    return data

if __name__ == "__main__":
    print(f'\ntraining model on {device}...\n')

    # data preprocessing
    data = load("../data/names.txt")
    words = [w.strip() for w in data.splitlines() if w]
    chars = sorted(list(set(''.join(words))))
    max_word_length = max(len(w) for w in words)

    # split data into training and test sets
    test_set_sz = min(1000, int(len(words) * 0.1)) # 10% of training set
    rp = torch.randperm(len(words)).tolist()
    train_words = [words[i] for i in rp[:-test_set_sz]]
    test_words = [words[i] for i in rp[-test_set_sz:]]
    print(f"split dataset into {len(train_words)} training examples and {len(test_words)} test examples")

    training_set = CharDataset(train_words, chars, max_word_length)
    test_set = CharDataset(test_words, chars, max_word_length)
    print("\nCOMPLETE: recurrent neural network\n")