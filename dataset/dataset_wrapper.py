import random
import torch
from torch.utils.data import Dataset, Subset
from autograder.dataset import CPEN455_2025_W1_Dataset


class CPEN455DatasetWrapper(CPEN455_2025_W1_Dataset):
    
    def __init__(self, csv_path=None):
        super().__init__(csv_path)
        
        for idx, label in enumerate(self._labels):
            if label == 'ham': 
                self._labels[idx] = 0
            elif label == 'spam':
                self._labels[idx] = 1
        
        self.num_spam = torch.tensor(self._index).sum()
        
    def get_num_spam(self):
        return self.num_spam
    
    def get_num_ham(self):
        return len(self) - self.get_num_spam()
        
def split_train_val(dataset, train_fraction, return_remaining=False):
    # Separate spam and ham samples
    spam_indices = [i for i, (_, _, _, label) in enumerate(dataset) if label == 1]
    ham_indices = [i for i, (_, _, _, label) in enumerate(dataset) if label == 0]

    # Ensure equal number of spam and ham samples
    num_spam = int(len(spam_indices) * train_fraction)
    num_ham = int(len(ham_indices) * train_fraction)
    
    selected_spam_indices = random.sample(spam_indices, num_spam)
    selected_ham_indices = random.sample(ham_indices, num_ham)

    # Combine and shuffle the selected indices
    selected_indices = selected_spam_indices + selected_ham_indices
    random.shuffle(selected_indices)

    if return_remaining:
        remaining_indices = list(set(range(len(dataset))) - set(selected_indices))
        return Subset(dataset, selected_indices), Subset(dataset, remaining_indices)
    else:
        return Subset(dataset, selected_indices)