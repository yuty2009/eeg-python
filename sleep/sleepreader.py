# -*- coding: utf-8 -*-

import os
import pyedflib
import torch
from torch.utils.data import Dataset
from common.datawrapper import read_matdata, read_gdfdata
from common.signalproc import *
from sleepstage import stage_dict


# translate epoch data (n, t, c) into grayscale images (n, 1, t, c)
class TransformEpoch(object):
    def __call__(self, epoch):
        epoch = torch.Tensor(epoch)
        return torch.unsqueeze(epoch, dim=0)


class EEGDataset(Dataset):
    def __init__(self, epochs, labels, transforms=None):
        if transforms == None:
            self.epochs = epochs
        else:
            self.epochs = [transforms(epoch) for epoch in epochs]
        self.labels = torch.Tensor(labels).long()

    def __getitem__(self, idx):
        return self.epochs[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


class SeqEEGDataset(Dataset):
    def __init__(self, epochs, labels, seqlen, transforms=None):
        if transforms == None:
            self.epochs = torch.Tensor(epochs)
        else:
            self.epochs = [transforms(epoch) for epoch in epochs]
            self.epochs = torch.stack(self.epochs)

        self.labels = torch.Tensor(labels).long()
        self.seqlen = seqlen
        assert self.seqlen <= len(self), "seqlen is too large"

    def __getitem__(self, idx):
        epoch_seq = torch.zeros(
            (self.seqlen,)+self.epochs.shape[1:], 
            dtype=self.epochs.dtype,
            device=self.epochs.device
            )
        idx1 = idx + 1
        if idx1 < self.seqlen:
            epoch_seq[-idx1:] = self.epochs[:idx1]
        else:
            epoch_seq = self.epochs[idx1-self.seqlen:idx1]
        return epoch_seq, self.labels[idx]

    def __len__(self):
        return len(self.labels) 


_available_dataset = [
    'sleep-edf-v1',
    'sleep-edf-ex',
    ]

# Have to manually define based on the dataset
ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3, "Sleep stage 4": 3, # Follow AASM Manual
    "Sleep stage R": 4,
    "Sleep stage ?": 6,
    "Movement time": 5
}


def load_eegdata(setname, datapath, subject):
    assert setname in _available_dataset, 'Unknown dataset name ' + setname
    if setname == 'sleepedf':
        filepath = os.path.join(datapath, subject+'.rec')
        labelpath = os.path.join(datapath, subject+'.hyp')
        data, target = load_eegdata_sleepedfx(filepath, labelpath)
    if setname == 'sleepedfx':
        filepath = os.path.join(datapath, subject+'-PSG.edf')
        labelpath = os.path.join(datapath, subject+'-Hypnogram.edf')
        data, target = load_eegdata_sleepedfx(filepath, labelpath)
    return data, target


def load_eegdata_sleepedf(rec_fname, hyp_fname):
    data = []
    target = []
    return data, target


def load_eegdata_sleepedfx(psg_fname, ann_fname, select_ch='EEG Fpz-Cz'):
    """
    https://github.com/akaraspt/tinysleepnet
    """
    
    psg_f = pyedflib.EdfReader(psg_fname)
    ann_f = pyedflib.EdfReader(ann_fname)

    assert psg_f.getStartdatetime() == ann_f.getStartdatetime()
    start_datetime = psg_f.getStartdatetime()

    file_duration = psg_f.getFileDuration()
    epoch_duration = psg_f.datarecord_duration
    if psg_f.datarecord_duration == 60: # Fix problems of SC4362F0-PSG.edf, SC4362FC-Hypnogram.edf
        epoch_duration = epoch_duration / 2

    # Extract signal from the selected channel
    ch_names = psg_f.getSignalLabels()
    ch_samples = psg_f.getNSamples()
    select_ch_idx = -1
    for s in range(psg_f.signals_in_file):
        if ch_names[s] == select_ch:
            select_ch_idx = s
            break
    if select_ch_idx == -1:
        raise Exception("Channel not found.")
    sampling_rate = psg_f.getSampleFrequency(select_ch_idx)
    n_epoch_samples = int(epoch_duration * sampling_rate)
    signals = psg_f.readSignal(select_ch_idx).reshape(-1, n_epoch_samples)

    # Sanity check
    n_epochs = psg_f.datarecords_in_file
    if psg_f.datarecord_duration == 60: # Fix problems of SC4362F0-PSG.edf, SC4362FC-Hypnogram.edf
        n_epochs = n_epochs * 2
    assert len(signals) == n_epochs, f"signal: {signals.shape} != {n_epochs}"

    # Generate labels from onset and duration annotation
    labels = []
    total_duration = 0
    ann_onsets, ann_durations, ann_stages = ann_f.readAnnotations()
    for a in range(len(ann_stages)):
        onset_sec = int(ann_onsets[a])
        duration_sec = int(ann_durations[a])
        ann_str = "".join(ann_stages[a])

        # Sanity check
        assert onset_sec == total_duration

        # Get label value
        label = ann2label[ann_str]

        # Compute # of epoch for this stage
        if duration_sec % epoch_duration != 0:
            raise Exception(f"Something wrong: {duration_sec} {epoch_duration}")
        duration_epoch = int(duration_sec / epoch_duration)

        # Generate sleep stage labels
        label_epoch = np.ones(duration_epoch, dtype=int) * label
        labels.append(label_epoch)

        total_duration += duration_sec

    labels = np.hstack(labels)

    # Remove annotations that are longer than the recorded signals
    labels = labels[:len(signals)]

    # Get epochs and their corresponding labels
    x = signals.astype(np.float32)
    y = labels.astype(np.int32)

    # Select only sleep periods
    w_edge_mins = 30
    nw_idx = np.where(y != stage_dict["W"])[0]
    start_idx = nw_idx[0] - (w_edge_mins * 2)
    end_idx = nw_idx[-1] + (w_edge_mins * 2)
    if start_idx < 0: start_idx = 0
    if end_idx >= len(y): end_idx = len(y) - 1
    select_idx = np.arange(start_idx, end_idx+1)
    x = x[select_idx]
    y = y[select_idx]

    # Remove movement and unknown
    move_idx = np.where(y == stage_dict["MOVE"])[0]
    unk_idx = np.where(y == stage_dict["UNK"])[0]
    if len(move_idx) > 0 or len(unk_idx) > 0:
        remove_idx = np.union1d(move_idx, unk_idx)
        select_idx = np.setdiff1d(np.arange(len(x)), remove_idx)
        x = x[select_idx]
        y = y[select_idx]

    # Save
    data_dict = {
        "x": x, 
        "y": y, 
        "fs": sampling_rate,
        "ch_label": select_ch,
        "start_datetime": start_datetime,
        "file_duration": file_duration,
        "epoch_duration": epoch_duration,
        "n_all_epochs": n_epochs,
        "n_epochs": len(x),
    }

    return data_dict


def extract_rawfeature(data, target, filter, sampleseg, chanset, standardize=True):
    num_trials, num_samples, num_channels = data.shape
    sample_begin = sampleseg[0]
    sample_end = sampleseg[1]
    num_samples_used = sample_end - sample_begin
    num_channel_used = len(chanset)

    # show_filtering_result(filter[0], filter[1], data[0,:,0])

    labels = target
    features = np.zeros([num_trials, num_samples_used, num_channel_used])
    for i in range(num_trials):
        signal_epoch = data[i]
        signal_filtered = signal_epoch
        for j in range(num_channels):
            signal_filtered[:, j] = signal.lfilter(filter[0], filter[1], signal_filtered[:, j])
            # signal_filtered[:, j] = signal.filtfilt(filter[0], filter[1], signal_filtered[:, j])
        if standardize:
            # init_block_size=1000 this param setting has a big impact on the result
            signal_filtered = exponential_running_standardize(signal_filtered, init_block_size=1000)
        features[i] = signal_filtered[sample_begin:sample_end, chanset]

    return features, labels


def load_npz_file(npz_file):
    """Load data and labels from a npz file."""
    with np.load(npz_file) as f:
        data = f["x"]
        labels = f["y"]
        sampling_rate = f["fs"]
    return data, labels, sampling_rate

def load_npz_list_files(npz_files):
    """Load data and labels from list of npz files."""
    data = []
    labels = []
    fs = None
    for npz_f in npz_files:
        print("Loading {} ...".format(npz_f))
        tmp_data, tmp_labels, sampling_rate = load_npz_file(npz_f)
        if fs is None:
            fs = sampling_rate
        elif fs != sampling_rate:
            raise Exception("Found mismatch in sampling rate.")

        # Reshape the data to match the input of the model - conv2d
        # tmp_data = np.squeeze(tmp_data)
        # tmp_data = tmp_data[:, :, np.newaxis, np.newaxis]
        
        # # Reshape the data to match the input of the model - conv1d
        tmp_data = tmp_data[:, :, np.newaxis]

        # Casting
        tmp_data = tmp_data.astype(np.float32)
        tmp_labels = tmp_labels.astype(np.int32)

        data.append(tmp_data)
        labels.append(tmp_labels)

    return data, labels

def load_subdata_preprocessed(datapath, subject):
    npz_f = os.path.join(datapath, subject+'.npz')
    data, labels, fs = load_npz_file(npz_f)
    return data, labels

def load_dataset_preprocessed(datapath, n_subjects=None):
    allfiles = os.listdir(datapath)
    npzfiles = []
    for idx, f in enumerate(allfiles):
        if ".npz" in f:
            npzfiles.append(os.path.join(datapath, f))
    npzfiles.sort()
    if n_subjects is not None:
        npzfiles = npzfiles[:n_subjects]
    subjects = [os.path.basename(npz_f)[:-4] for npz_f in npzfiles]
    data, labels = load_npz_list_files(npzfiles)
    return data, labels, subjects


if __name__ == '__main__':

    import os
    import glob

    datapath = '/Users/yuty2009/data/eegdata/sleep/sleep-edf-database-expanded-1.0.0/sleep-cassette/'
    if not os.path.isdir(datapath + 'processed/'):
        os.mkdir(datapath + 'processed/')

    psg_fnames = glob.glob(os.path.join(datapath, "*PSG.edf"))
    ann_fnames = glob.glob(os.path.join(datapath, "*Hypnogram.edf"))
    psg_fnames.sort()
    ann_fnames.sort()
    psg_fnames = np.asarray(psg_fnames)
    ann_fnames = np.asarray(ann_fnames)

    for i in range(len(psg_fnames)):

        subject = os.path.basename(psg_fnames[i])[:-8]

        print('Load and extract continuous EEG into epochs for subject '+subject)
        data_dict = load_eegdata_sleepedfx(psg_fnames[i], ann_fnames[i])

        np.savez(datapath+'processed/'+subject+'.npz', **data_dict)


