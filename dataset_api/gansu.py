from seisbench.data.base import WaveformDataset


class Gansu(WaveformDataset):

    def __init__(self, path="/data/djx/seisbench_cache/datasets/gansu", chunks=None, sampling_rate=100, **kwargs):
        citaion = ("Ye, L., 2023. Research and design of earthquake detection and phase picking based on "
                   "transfer learning and attention mechanisms. master’s thesis. Lanzhou University.")
        super().__init__(path=path, name="Gansu", chunks=chunks, sampling_rate=sampling_rate, **kwargs)
