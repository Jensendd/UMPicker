from seisbench.data.base import WaveformDataset


class DiTing(WaveformDataset):

    def __init__(self, path="/data/djx/seisbench_cache/datasets/diting", chunks=None, sampling_rate=50, **kwargs):
        citation = (
            "Zhao, M., Xiao, Z., Chen, S., Fang, L., 2023. Diting: A large-scale chinese seismic benchmark dataset "
            "for artificial intelligence in seismology. Earthquake Science 36, 84–94.")
        super().__init__(path=path, name="DiTing", chunks=chunks, sampling_rate=sampling_rate, **kwargs)
