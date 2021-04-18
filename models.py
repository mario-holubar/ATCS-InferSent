import torch
from torch import nn
import pytorch_lightning as pl
from data import TEXT

# Sentence encoder that simply averages word embeddings
class MeanEncoder(torch.nn.Module):
    def __init__(self, word_embeddings):
        super(MeanEncoder, self).__init__()
        self.embed = nn.Embedding.from_pretrained(word_embeddings, freeze=True, padding_idx=1)
    
    def forward(self, x):
        embeddings = self.embed(x)
        return embeddings.mean(1)

# Sentence encoder using unidirectional LSTM
class LSTMEncoder(torch.nn.Module):
    def __init__(self, word_embeddings, hidden_dim):
        super(LSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding.from_pretrained(word_embeddings, freeze=True, padding_idx=1)
        self.lstm = nn.LSTM(300, hidden_dim, 1, batch_first=True)
    
    def forward(self, x):
        embeddings = self.embed(x)
        _, (h, _) = self.lstm(embeddings)
        return h.view(-1, self.hidden_dim)

# Sentence encoder using unidirectional LSTM
class BiLSTMEncoder(torch.nn.Module):
    def __init__(self, word_embeddings, hidden_dim):
        super(BiLSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding.from_pretrained(word_embeddings, freeze=True, padding_idx=1)
        self.lstm = nn.LSTM(300, hidden_dim // 2, 1, bidirectional=True, batch_first=True)
    
    def forward(self, x):
        embeddings = self.embed(x)
        _, (h, _) = self.lstm(embeddings)
        h = h.view(2, -1, self.hidden_dim // 2)
        return torch.cat([h[0], h[1]], 1)

# Sentence encoder using unidirectional LSTM
class PooledBiLSTMEncoder(torch.nn.Module):
    def __init__(self, word_embeddings, hidden_dim):
        super(PooledBiLSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding.from_pretrained(word_embeddings, freeze=True, padding_idx=1)
        self.lstm = nn.LSTM(300, hidden_dim // 2, 1, bidirectional=True, batch_first=True)
    
    def forward(self, x):
        embeddings = self.embed(x)
        h, _ = self.lstm(embeddings)
        return h.max(1).values



# Classifier that takes premise and hypothesis sentence encodings and predicts entailment
class Classifier(pl.LightningModule):
    def __init__(self, args):
        super(Classifier, self).__init__()

        self.save_hyperparameters()

        if args.encoder == 'mean': self.encoder = MeanEncoder(TEXT.vocab.vectors)
        elif args.encoder == 'lstm': self.encoder = LSTMEncoder(TEXT.vocab.vectors, args.sentence_dim)
        elif args.encoder == 'bilstm': self.encoder = BiLSTMEncoder(TEXT.vocab.vectors, args.sentence_dim)
        elif args.encoder == 'pooledbilstm': self.encoder = PooledBiLSTMEncoder(TEXT.vocab.vectors, args.sentence_dim)
        else:
            raise ValueError("Encoder must be one of ['mean', 'lstm', 'bilstm', 'pooledbilstm']!")
        
        if args.encoder == 'mean':
            self.fc = nn.Sequential(nn.Linear(4 * 300, 512), nn.Tanh(), nn.Linear(512, 512), nn.Tanh(), nn.Linear(512, 3))
        else:
            self.fc = nn.Sequential(nn.Linear(4 * args.sentence_dim, 512), nn.Tanh(), nn.Linear(512, 512), nn.Tanh(), nn.Linear(512, 3))

        self.loss = torch.nn.CrossEntropyLoss()
        self.accuracy = pl.metrics.Accuracy()

        self.last_acc = 0
        self.current_acc = 0
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        lr_decay = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
        return [optimizer], [lr_decay]
    
    def forward(self, premise, hypothesis):
        premise = self.encoder(premise)
        hypothesis = self.encoder(hypothesis)
        diff = (premise - hypothesis).abs()
        mul = premise * hypothesis
        features = torch.cat([premise, hypothesis, diff, mul], 1)
        return self.fc(features)
    
    def training_step(self, batch, batch_idx):
        y = self(batch.premise, batch.hypothesis)
        l = self.loss(y, batch.label)
        pred = y.argmax(1)
        acc = self.accuracy(pred, batch.label)
        self.log('train_loss', l)
        self.log('train_acc', acc, prog_bar=True)
        return l
    
    def validation_step(self, batch, batch_idx):
        y = self(batch.premise, batch.hypothesis)
        l = self.loss(y, batch.label)
        pred = y.argmax(1)
        acc = self.accuracy(pred, batch.label)
        self.log('valid_loss', l)
        self.log('valid_acc', acc)
        return acc # TODO pred == batch.label?
    
    def test_step(self, batch, batch_idx):
        y = self(batch.premise, batch.hypothesis)
        l = self.loss(y, batch.label)
        pred = y.argmax(1)
        acc = self.accuracy(pred, batch.label)
        self.log('test_loss', l)
        self.log('test_acc', acc)
        return acc
    
    # Decrease learning rate by a factor of 5 if validation accuracy improved
    def validation_epoch_end(self, acc):
        self.last_acc = self.current_acc
        self.current_acc = sum(acc) / len(acc)
        if self.current_acc < self.last_acc:
            for pg in self.trainer.optimizers[0].param_groups:
                pg['lr'] *= 0.2
    
    # Stop training if learning rate goes below 1e-5
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        for pg in self.trainer.optimizers[0].param_groups:
            if pg['lr'] < 0.00001: return -1