from __future__ import absolute_import, division, unicode_literals

import io
import os.path as osp
from typing import Callable, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from mpixel.analysis.splitclassifier import SplitClassifier


# A batcher maps a list of input strings to a dict {layer_idx: (B, D) ndarray}
# of mean-pooled hidden states, mirroring the original LINSPECTOR ``batcher``.
Batcher = Callable[[List[str]], Dict[int, np.ndarray]]


class Probe:
    """LINSPECTOR intrinsic (type-level) probing for a single (task, language).

    Loads the ``train``/``dev``/``test`` splits for one probing task, extracts
    per-layer embeddings through a supplied ``batcher`` callable, and trains a
    logistic-regression probe per layer (with a regularization sweep on dev).
    """

    def __init__(self, task, lang, data_dir, seed=1111):
        self.seed = seed
        self.task = task
        self.lang = lang
        self.task_data = {'train': {'X': [], 'y': []},
                          'dev': {'X': [], 'y': []},
                          'test': {'X': [], 'y': []}}

        self.loadData(data_dir, lang, task)

    def loadData(self, data_dir, lang, task):
        splits = ['train', 'dev', 'test']
        for split in splits:
            file_path = osp.join(osp.abspath(osp.expanduser(data_dir)), task, lang, f'{split}.txt')
            if not osp.isfile(file_path):
                raise FileNotFoundError("File does not exist at {}".format(file_path))
            with io.open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.rstrip().split()
                    self.task_data[split]['X'].append(' '.join(line[0:-1]))
                    self.task_data[split]['y'].append(line[-1])

        labels = sorted(np.unique(self.task_data['train']['y']))
        self.tok2label = dict(zip(labels, range(len(labels))))
        self.nclasses = len(self.tok2label)

        for split in self.task_data:
            for i, y in enumerate(self.task_data[split]['y']):
                try:
                    self.task_data[split]['y'][i] = self.tok2label[y]
                except:
                    print(y)
                    print(self.task_data[split]['X'][i])
                    quit()

    def run(self, batcher: Batcher, batch_size: int = 32, layers: Optional[List[int]] = None):
        task_embed = {'train': {}, 'dev': {}, 'test': {}}
        bsize = batch_size

        print(f'Computing embeddings for train/dev/test for {self.task}')

        for key in self.task_data:
            indexes = list(range(len(self.task_data[key]['y'])))
            layer_embs: Dict[int, list] = {}

            sorted_data = sorted(zip(self.task_data[key]['X'],
                                     self.task_data[key]['y'], indexes),
                                 key=lambda z: (len(z[0]), z[1], z[2]))

            self.task_data[key]['X'], self.task_data[key]['y'], self.task_data[key]['idx'] = map(list,
                                                                                                 zip(*sorted_data))

            task_embed[key]['X'] = {}

            for i in tqdm(range(0, len(self.task_data[key]['y']), bsize)):
                batch = self.task_data[key]['X'][i:i + bsize]
                embs = batcher(batch)
                for k, v in embs.items():
                    layer_embs.setdefault(k, []).append(v)
            for layer, chunks in layer_embs.items():
                task_embed[key]['X'][layer] = np.vstack(chunks)
            task_embed[key]['y'] = np.array(self.task_data[key]['y'])
            task_embed[key]['idx'] = np.array(indexes)
        print("Computed embeddings!")

        # Default to every transformer layer (skip the embedding output at 0),
        # discovered dynamically so this works for models of any depth.
        if layers is None:
            layers = [layer for layer in sorted(task_embed['train']['X']) if layer >= 1]

        first_layer = layers[0]
        assert task_embed['train']['X'][first_layer].shape[0] == task_embed['train']['y'].shape[0] == \
               task_embed['train']['idx'].shape[0]

        params_classifier = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                             'tenacity': 5, 'epoch_size': 4}
        config_classifier = {'nclasses': self.nclasses, 'seed': 1223,
                             'usepytorch': True,
                             'classifier': params_classifier}

        results = {}
        for layer in tqdm(layers):
            print(f"Training classifier on embeddings from layer {layer}...")
            clf = SplitClassifier(X={'train': task_embed['train']['X'][layer],
                                     'valid': task_embed['dev']['X'][layer],
                                     'test': task_embed['test']['X'][layer]},
                                  y={'train': task_embed['train']['y'],
                                     'valid': task_embed['dev']['y'],
                                     'test': task_embed['test']['y']},
                                  config=config_classifier)

            devacc, testacc, predictions = clf.run()
            results[layer] = (devacc, testacc, predictions)
            print(f"Dev acc : {devacc} Test acc : {testacc} on {layer} for {self.task}")

        return results
