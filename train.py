import torch
import torchtext
import pytorch_lightning as pl
from data import train_data, valid_data, test_data
import models

def train(args):
    if args.debug:
        train_dataloader = torchtext.data.BucketIterator(test_data, batch_size=64, train=True)
    else:
        train_dataloader = torchtext.data.BucketIterator(train_data, batch_size=64, train=True)
    valid_dataloader = torchtext.data.BucketIterator(valid_data, batch_size=64, train=False)

    classifier = models.Classifier(args)

    logger = pl.loggers.TensorBoardLogger('logs', args.encoder)
    lr_monitor = pl.callbacks.LearningRateMonitor('epoch')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='valid_loss', dirpath=f"checkpoints/{args.encoder}", filename=f"version_{logger.version}")

    trainer = pl.Trainer(max_epochs=args.max_epochs,
                        logger=[logger],
                        callbacks=[lr_monitor, checkpoint_callback],
                        gpus=torch.cuda.device_count(),
                        progress_bar_refresh_rate=args.progress_bar,
                        weights_summary=None)

    print("Starting training...")
    trainer.fit(classifier, train_dataloader, valid_dataloader)
    print("Done training!")

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('encoder', choices=['mean', 'lstm', 'bilstm', 'pooledbilstm'], type=str.lower, help="Which encoder architecture to use.")
    parser.add_argument('--sentence_dim', type=int, default=1024, help="Size of a single sentence embedding.")
    parser.add_argument('--max_epochs', type=int, default=20, help="Maximum number of epochs to train.")

    parser.add_argument('--progress_bar', type=int, default=0, help="Progress bar refresh rate. 0 for off.")
    parser.add_argument('--debug', action='store_true', help="Use small dataset for testing")

    args = parser.parse_args()
    train(args)