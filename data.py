import torch
import torchtext
import spacy
import os
import pickle

# To prevent deprecation warnings from torchtext 0.7 (which is required for SNLI)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

if not spacy.util.is_package("en_core_web_sm"):
    print("Downloading spaCy model...")
    import spacy.cli
    spacy.cli.download('en_core_web_sm')

print("Loading data...")
TEXT = torchtext.data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm', batch_first=True)
LABEL = torchtext.data.Field(sequential=False, is_target=True, unk_token=None)
train_data, valid_data, test_data = torchtext.datasets.nli.SNLI.splits(TEXT, LABEL, root = 'data')

if os.path.exists("vocab.pkl"):
    print("Loading vocabulary...")
    with open("vocab.pkl", 'rb') as f:
        TEXT.vocab = pickle.load(f)
else:
    print("Building vocabulary...")
    TEXT.build_vocab(train_data, vectors=torchtext.vocab.GloVe(name='840B', dim=300))
    with open("vocab.pkl", 'wb') as f:
        pickle.dump(TEXT.vocab, f)
LABEL.build_vocab(train_data)

# Find word from vector (debug / sanity check)
'''glove_lengths = torch.sqrt((TEXT.vocab.vectors ** 2).sum(dim=1))
def closest_cosine(vec):
    numerator = (TEXT.vocab.vectors * vec).sum(dim=1)
    denominator = glove_lengths * torch.sqrt((vec ** 2).sum())
    similarities = numerator / denominator
    similarities[similarities != similarities] = 0
    return TEXT.vocab.itos[similarities.argmax()]'''