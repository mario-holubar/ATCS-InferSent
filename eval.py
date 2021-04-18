import torch
import torchtext
import pytorch_lightning as pl
import os

from data import TEXT, LABEL, test_data
import models



def eval_SNLI(classifier):
    test_dataloader = torchtext.data.BucketIterator(test_data, batch_size=64, train=False)

    logger = pl.loggers.TensorBoardLogger('logs', encoder, version=version)

    trainer = pl.Trainer(logger=[logger],
                        gpus=torch.cuda.device_count(),
                        progress_bar_refresh_rate=0,
                        weights_summary=None)

    print(trainer.test(classifier, test_dataloader))



def eval_SentEval(classifier):
    import sys
    import logging

    PATH_TO_SENTEVAL = 'SentEval-master/'
    PATH_TO_DATA = 'SentEval-master/data'
    sys.path.insert(0, PATH_TO_SENTEVAL)
    import senteval

    def batcher(params, batch):
        x = TEXT.process(TEXT.pad(batch))
        return classifier.encoder(x).detach()

    params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
    params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                    'tenacity': 5, 'epoch_size': 4}
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

    se = senteval.engine.SE(params, batcher)
    transfer_tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                    'SICKEntailment', 'SICKRelatedness', 'STSBenchmark']
    results = se.eval(transfer_tasks)
    print(results)



if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('encoder', choices=['mean', 'lstm', 'bilstm', 'pooledbilstm'], type=str.lower, help="Which encoder architecture to use.")
    parser.add_argument('--version', type=int, help="Which version (checkpoint) to use. Defaults to latest.")
    args = parser.parse_args()
    encoder = args.encoder
    version = args.version

    # If no version specified, use the latest model
    if version is None:
        versions = [int(f[8:-5]) for f in os.listdir(f"checkpoints/{encoder}")]
        version = versions[-1]

    model_path = f"checkpoints/{encoder}/version_{version}.ckpt"
    print("Using " + model_path)
    classifier = models.Classifier.load_from_checkpoint(model_path)
    
    eval_SentEval(classifier)
    eval_SNLI(classifier)