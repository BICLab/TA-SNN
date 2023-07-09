import os, h5py
import bisect, random
import numpy as np
from torch.utils.data import Dataset


class SHD_Dataset(Dataset):

    def __init__(
            self,
            path,
            train=True,
            input_length=700,
            T=100,
            dt=1.4 / 100,
            ds=1,
            is_train_Enhanced=False,
            clips=1,
    ):
        super(SHD_Dataset, self).__init__()
        self.train = train
        self.input_length = input_length
        self.T = T
        self.dt = dt
        self.ds = ds
        self.is_train_Enhanced = is_train_Enhanced
        self.clips = clips

        if self.train:
            self.fileName = 'shd_train.h5'
        else:
            self.fileName = 'shd_test.h5'

        file = h5py.File(os.path.join(path, self.fileName), 'r')

        X = file['spikes']

        self.label = file['labels']
        self.firing_times = X['times']
        self.units_fired = X['units']

    def __len__(self):
        return self.label.size

    def __getitem__(self, idx):
        if self.train:
            data = np.stack((self.firing_times[idx], self.units_fired[idx])).T
            if self.is_train_Enhanced:
                if self.firing_times[idx].max() - self.firing_times[idx].min() > self.T * self.dt:
                    start_time = random.uniform(
                        0,
                        self.firing_times[idx].max() - self.firing_times[idx].min() - self.T * self.dt
                    )
                    idx_start = find_first(data[:, 0], start_time)
                    data = data[idx_start:, :]

            data = shd_evs_pol_dvs(
                data=data,
                dt=self.dt,
                T=self.T,
                input_length=self.input_length,
                ds=self.ds,
            )

            label_idx = self.label[idx]
            label = np.zeros((20))
            label[label_idx] = 1.0

            return data, label
        else:
            data = np.stack((self.firing_times[idx], self.units_fired[idx])).T
            data = sample_test(
                data=data,
                T=self.T,
                clips=self.clips,
                dt=self.dt,
            )
            label_idx = self.label[idx]
            label = np.zeros((20))
            label[label_idx] = 1.0
            data_temp = []
            target_temp = []
            for i in range(self.clips):
                temp = shd_evs_pol_dvs(
                    data=data[i],
                    dt=self.dt,
                    T=self.T,
                    input_length=self.input_length,
                    ds=self.ds,
                )
                data_temp.append(temp)
                target_temp.append(label)
            data = np.array(data_temp)
            label = np.array(target_temp)

            return data, label


def sample_test(data,
                T=60,
                clips=1,
                dt=0.5
                ):
    data[:, 0] -= data[0, 0]

    start_time = data[0, 0]
    end_time = data[-1, 0]

    start_point = []
    if clips * T * dt - (end_time - start_time) > 0:
        overlap = int(
            np.floor((clips * T * dt - (end_time - start_time)) / clips))
        for j in range(clips):
            start_point.append(j * (T * dt - overlap))
            if start_point[-1] + T * dt > end_time:
                diff = start_point[-1] + T * dt - end_time
                start_point[-1] = start_point[-1] - diff
    else:
        overlap = int(
            np.floor(((end_time - start_time) - clips * T * dt) / clips))
        for j in range(clips):
            start_point.append(j * (T * dt + overlap))
            if start_point[-1] + T * dt > end_time:
                diff = start_point[-1] + T * dt - end_time
                start_point[-1] = start_point[-1] - diff

    temp = []
    for start in start_point:
        idx_beg = find_first(data[:, 0], start)
        idx_end = find_first(data[:, 0][idx_beg:], start + T * dt) + idx_beg
        temp.append(data[idx_beg:idx_end])

    return temp


def shd_evs_pol_dvs(data, dt=1.4, T=100, input_length=700, ds=1):
    t_start = data[0][0]
    ts = np.linspace(t_start, t_start + T * dt, num=T)
    chunks = np.zeros([T] + [input_length // ds], dtype='int64')

    idx_start = 0
    idx_end = 0
    for i, t in enumerate(ts):
        idx_end += find_first(data[idx_end:, 0], t + dt)
        if idx_end > idx_start:
            ee = data[idx_start:idx_end, 1:]
            pol = ee[:, 0].astype(np.int)
            np.add.at(chunks, (i, pol), 1)

        idx_start = idx_end

    return chunks


def find_first(a, tgt):
    return bisect.bisect_left(a, tgt)

