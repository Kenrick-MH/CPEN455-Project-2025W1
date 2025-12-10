import random

from bidict import bidict
import pandas as pd
from torch.utils.data import Dataset, Subset

class Enron1Dataset(Dataset):
    def __init__(
        self,
        csv_path = None,
        n_samples = None
    ) -> None:
    
        self._csv_path = csv_path
        frame = pd.read_csv(csv_path, index_col=0)   
        
        if n_samples is not None:
            frame = frame[:n_samples]
             
        print(f' Total number of data {len(frame)}')
             
        frame["text"] = frame["text"].fillna("").astype(str)
        frame["label_num"] = (
            frame["label_num"].fillna("").astype(str).str.strip().str.lower()
        )

        self._index = frame.index.to_list()
        self._msgs = frame["text"].to_list()
        self._labels = frame["label_num"].tolist()

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(
        self, index
    ):
        data_index = self._index[index]
        message = self._msgs[index]
        label_index = int(self._labels[index])

        return data_index, message, label_index

    def prepare_subsets(self, train_ratio=0.8, return_remaining=False):
        # Separate spam and ham samples
        spam_indices = [i for i, (_, _, label) in enumerate(self) if label == 1]
        ham_indices = [i for i, (_, _, label) in enumerate(self) if label == 0]

        
        num_spam_train = int(train_ratio * len(spam_indices))
        num_ham_train = int(train_ratio * len(ham_indices))
        selected_spam_indices = random.sample(spam_indices, num_spam_train)
        selected_ham_indices = random.sample(ham_indices, num_ham_train)

        print(f'Train distribution: S {len(selected_spam_indices)} H{len(selected_ham_indices)}')

        # Combine and shuffle the selected indices
        selected_indices = selected_spam_indices + selected_ham_indices
        random.shuffle(selected_indices)

        if return_remaining:
            remaining_indices = list(set(range(len(self))) - set(selected_indices))
            return Subset(self, selected_indices), Subset(self, remaining_indices)
        else:
            return Subset(self, selected_indices)
