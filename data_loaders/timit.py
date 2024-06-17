# data_loaders/timit.py

import torchaudio
# from torchaudio.datasets import TIMIT
# from torch.utils.data import DataLoader
#
def load_timit(batch_size=64):
    train_dataset = TIMIT(root='./data', download=True, subset='train')
    test_dataset = TIMIT(root='./data', download=True, subset='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

import torchaudio
# from torchaudio.datasets.utils import download_url, extract_archive
from torchtext.utils import download_from_url, extract_archive
from torch.hub import download_url_to_file

from pathlib import Path

class TIMITDataset(torchaudio.datasets.SPEECHCOMMANDS):
    def __init__(self, root, download=False):
        url = "https://data.deepai.org/timit.zip"  # Replace with the actual URL
        archive = Path(root) / "timit.zip"
        folder = Path(root) / "TIMIT"

        if download and not folder.exists():
            download_from_url(url, root)
            extract_archive(str(archive), str(root))

        super().__init__(root=root, download=download)

    def __getitem__(self, index):
        # Implement the logic to read and return a data sample
        # For example, read the audio file and label
        pass

    def __len__(self):
        # Return the number of samples in the dataset
        pass

# Example usage
root = "./data"
timit_dataset = TIMITDataset(root, download=True)
