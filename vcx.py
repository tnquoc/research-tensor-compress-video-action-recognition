import os

from numpy.ma import copy
from torchvision import transforms
from torch.utils.data import DataLoader

import wfdb
from transforms import ToTensor, Resample, ApplyGain
import lorem
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset

from vocab import Vocabulary
from dataset import collate_fn
import librosa
from preprocess import bwr_dwt, norm
from sklearn import preprocessing


class RealDataset(Dataset):
    def __init__(self, length, topic, vocab, train, waveform_dir, in_length, dataset, transform, in_channels=3,
                 label='Label'):
        self.topic = topic
        self.dataset = dataset
        self.waveform_dir = waveform_dir
        self.in_length = in_length
        self.length = length
        self.transform = transform
        self.in_channels = in_channels
        self.label = label
        self.tokenizer = RegexpTokenizer('\d+\.?,?\d+|-?/?\w+-?/?\w*|\w+|\d+|<[A-Z]+>')
        self.weights = self.setup_weights(self.dataset['Label'])
        if train:
            self.vocab = self.setup_vocab(self.dataset['Label'])
        else:
            self.vocab = vocab
        self.train = train

    def setup_vocab(self, labels, threshold=1):
        corpus = labels.str.cat(sep=" ")

        counter = Counter(self.tokenizer.tokenize(corpus))
        del counter['']

        counter = counter.most_common()
        words = []
        cnts = []
        for i in range(0, len(counter)):
            words.append(counter[i][0])
            if words[-1] in ['artifact', 'sinus', 'arrhythmia', 'bradycardia', 'rhythm']:
                cnts.append(0.25)
            elif words[-1] in ['atrial', 'fibrillation', 'flutter', 'supraventricular', 'tachycardia', 'paroxysmal',
                               'ventricular', 'run', '1st', 'degree', '2nd', '3rd', 'advanced', 'heart', 'block',
                               'high', 'grade', 'ivcd', 'pause', 'urgent', 'emergent', 'svt', 'psvt',
                               'fibrillation/flutter', 'av']:
                cnts.append(0.9)
            else:
                cnts.append(0.1)
        vocab = Vocabulary()
        vocab.add_word('<pad>', min(cnts))
        vocab.add_word('<start>', min(cnts))
        vocab.add_word('<end>', min(cnts))
        vocab.add_word('<unk>', min(cnts))
        # Add the words to the vocabulary.
        for i, word in enumerate(words):
            vocab.add_word(word, cnts[i])

        return vocab

    def setup_weights(self, labels):
        weights = np.zeros(len(labels))
        for i in range(0, len(labels)):
            if 'fibrillation' in labels[i] or 'flutter' in labels[i] or 'atrial run' in labels[i] \
                    or 'supraventricular tachycardia' in labels[i] or 'vt' in labels[i] \
                    or 'ventricular tachycardia' in labels[i] or 'ventricular run' in labels[i] or 'ivcd' in labels[i] \
                    or '1st' in labels[i] or '2nd' in labels[i] or '3rd' in labels[i] or \
                    'advanced heart block' in labels[i] or 'high grade heart block' in labels[i] \
                    or 'pause' in labels[i] or 'urgent' in labels[i] or 'emergent' in labels[i] \
                    or 'accelerated' in labels[i] or 'av' in labels[i] or 'pace' in labels[i] \
                    or 'atrial tachycardia' in labels[i] or 'variable block' in labels[i]:
                weights[i] = 0.75
            elif 'bigeminy' in labels[i] or 'trigeminy' in labels[i] or 'quadrigeminy' in labels[i] \
                    or 'pac' in labels[i] or 've' in labels[i] or 'pvc' in labels[i] or 'rb' in labels[i]:
                weights[i] = 0.5
            elif 'artifact' in labels[i] or 'sinus' in labels[i] or 'ectopic' in labels[i] or 'junctional' in labels[i] \
                    or 'no enough' in labels[i]:
                weights[i] = 0.25
            else:
                raise ValueError(f'Error at labels {labels[i]} {i}')

        return weights

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        waveform, spec, sample_id = self.get_waveform(idx, self.in_channels)

        sample = {
            'waveform': waveform,
            'spec': spec,
            'id': sample_id,
            'weights': self.weights[idx]
        }

        if self.label in self.dataset.columns.values:
            sentence = self.dataset[self.label].iloc[idx]
            try:
                tokens = self.tokenizer.tokenize(sentence)
            except:
                print(sentence)
                raise Exception()
            vocab = self.vocab
            caption = [vocab('<start>')]
            caption.extend([vocab(token) for token in tokens])
            caption.append(vocab('<end>'))
            target = torch.Tensor(caption)

            sample['label'] = target

        if self.topic:
            topic_label_classes = 100

            topic_labels_bools = np.random.randint(2, size=topic_label_classes)
            topic_tensor = torch.from_numpy(topic_labels_bools).float()
            topic_tensor_norm = topic_tensor / topic_tensor.sum()

            sample['extra_label'] = topic_tensor_norm

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_waveform(self, idx, in_channels):
        try:
            raw_signal, fields = wfdb.rdsamp(os.path.join(self.waveform_dir, self.dataset['Path'][idx])
                                             + '/' + self.dataset['PseudoID'][idx])
        except:
            raw_signal, fields = wfdb.rdsamp(self.waveform_dir + '/' + self.dataset['PseudoID'][idx])

        if in_channels == 1:
            channel = self.dataset['Channel'][idx]
            waveform = np.array([raw_signal[:, channel]])
        else:
            waveform = np.array(raw_signal.T)

        waveform = np.nan_to_num(waveform)
        # waveform = bwr_dwt(waveform[0, :], fields['fs'])
        # waveform = np.where(waveform > 20, 20, waveform)
        # waveform = np.where(waveform < -20, -20, waveform)
        # waveform = np.expand_dims(waveform, axis=0)
        spec = np.squeeze(waveform)
        # spec = np.abs(librosa.stft(spec, hop_length=512))
        spec = librosa.feature.melspectrogram(y=spec, sr=250)
        # spec = librosa.amplitude_to_db(spec, ref=np.max) / 255
        spec = librosa.power_to_db(spec, ref=np.max)
        # spec = (spec - np.mean(spec, axis=0)) / np.std(spec, axis=0)
        spec = librosa.util.normalize(spec)

        # waveform = waveform.T
        spec = np.expand_dims(spec, axis=0)
        if 'IsAugment' in self.dataset and self.dataset['IsAugment'][idx]:
            # f = np.random.randint(1, 3)
            # f0 = np.random.randint(128 - f + 1)
            # spec[:, f0:f0+f, :] = np.zeros((f, 30)) + np.max(spec)
            c_o_r = np.random.randint(1, 4)
            if c_o_r == 1:
                f = np.random.randint(1, 4)
                f0 = np.random.randint(30 - f + 1)
                spec[:, :, f0:f0+f] = np.zeros((128, f)) + np.max(spec)
            elif c_o_r == 2:
                f = np.random.randint(1, 4)
                f0 = np.random.randint(128 - f + 1)
                spec[:, f0:f0+f, :] = np.zeros((f, 30)) + np.max(spec)
            else:
                f = np.random.randint(1, 4)
                f0 = np.random.randint(30 - f + 1)
                spec[:, :, f0:f0 + f] = np.zeros((128, f)) + np.max(spec)
                f = np.random.randint(1, 4)
                f0 = np.random.randint(128 - f + 1)
                spec[:, f0:f0 + f, :] = np.zeros((f, 30)) + np.max(spec)

        return waveform, torch.from_numpy(spec).type(torch.FloatTensor), idx


class FakeDataset:
    def __init__(self, length, topic, vocab, transform):
        self.length = length
        self.topic = topic
        self.transform = transform
        self.tokenizer = RegexpTokenizer('\d+\.?,?\d+|-?/?\w+-?/?\w*|\w+|\d+|<[A-Z]+>')

        if vocab is None:
            self.vocab = self.setup_vocab(0)
        else:
            self.vocab = vocab

    def setup_vocab(self, threshold):
        corpus = " ".join([lorem.sentence() for _ in range(self.length)])

        counter = Counter(self.tokenizer.tokenize(corpus))
        del counter['']

        words = [word for word, cnt in counter.items() if cnt >= threshold]

        vocab = Vocabulary()
        vocab.add_word('<pad>')
        vocab.add_word('<start>')
        vocab.add_word('<end>')
        vocab.add_word('<unk>')

        # Add the words to the vocabulary.
        for _, word in enumerate(words):
            vocab.add_word(word)
        return vocab

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        waveform = np.random.rand(12, 5000)

        sample = {
            'waveform': waveform,
            'samplebase': 500,
            'gain': 4.88,
            'id': int(np.random.rand(1)),
        }

        sentence = lorem.sentence()

        try:
            tokens = self.tokenizer.tokenize(sentence)
        except:
            print(sentence)
            raise Exception()

        vocab = self.vocab
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)

        sample['label'] = target

        if self.topic:
            topic_label_classes = 100

            topic_labels_bools = np.random.randint(2, size=topic_label_classes)
            topic_tensor = torch.from_numpy(topic_labels_bools).float()
            topic_tensor_norm = topic_tensor / topic_tensor.sum()

            sample['extra_label'] = topic_tensor_norm

        if self.transform:
            sample = self.transform(sample)

        return sample


def get_loaders(params, topic):
    transform = transforms.Compose([ToTensor()])

    train_df = pd.read_csv(params['train_labels_csv'])
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    val_df = pd.read_csv(params['val_labels_csv'])
    val_df = val_df.sample(frac=1).reset_index(drop=True)

    is_train, vocab = True, None
    trainset = RealDataset(len(train_df), topic, vocab, is_train, params['data_dir'], params['in_length'], train_df,
                           transform=transform, in_channels=params['in_channels'])

    is_train, vocab = False, trainset.vocab
    valset = RealDataset(len(val_df), topic, vocab, is_train, params['data_dir'], params['in_length'],
                         val_df, transform=transform, in_channels=params['in_channels'])

    testset_df = pd.read_csv(params['test_labels_csv'])
    testset = RealDataset(len(testset_df), topic, vocab, is_train, params['data_dir'], params['in_length'],
                          testset_df, transform=transform, in_channels=params['in_channels'])

    train_loader = DataLoader(trainset, batch_size=params['batch_size'],
                              num_workers=4, collate_fn=collate_fn, shuffle=True)

    val_loader = DataLoader(valset, batch_size=params['batch_size'],
                            num_workers=4, collate_fn=collate_fn)

    test_loader = DataLoader(testset, batch_size=params['batch_size'],
                             num_workers=4, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, vocab


def get_loaders_toy_data(params, topic=False):
    vocab = None
    transform = transforms.Compose([Resample(500), ToTensor(), ApplyGain()])
    train_set = FakeDataset(1000, topic, vocab, transform)
    vocab = train_set.vocab
    val_set = FakeDataset(200, topic, vocab, transform)
    test_set = FakeDataset(200, topic, vocab, transform)

    train_loader = DataLoader(train_set, batch_size=params['batch_size'],
                              num_workers=4, collate_fn=collate_fn, shuffle=True)

    val_loader = DataLoader(val_set, batch_size=params['batch_size'],
                            num_workers=4, collate_fn=collate_fn)

    test_loader = DataLoader(test_set, batch_size=params['batch_size'],
                             num_workers=4, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, vocab
