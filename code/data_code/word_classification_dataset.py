import torch
from torch.utils.data import Dataset

class WordClassificationDataset(Dataset):
    def __init__(self, base_dataset, word_to_idx):
        self.base_dataset = base_dataset
        self.word_to_idx = word_to_idx

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        try:
            # von IAMWordDataset (liefert image, encoded, length)
            batch_data = self.base_dataset[idx]
            if batch_data is None:
                return None
                
            image, _, _ = batch_data 
            
            # text finden
            actual_ds = self.base_dataset
            actual_idx = idx
            
            # falls Subset ist (random_split), muss tiefer 
            if isinstance(self.base_dataset, torch.utils.data.Subset):
                actual_ds = self.base_dataset.dataset
                actual_idx = self.base_dataset.indices[idx]

            if hasattr(actual_ds, 'samples'):
                _, word = actual_ds.samples[actual_idx]
            else:
                word = actual_ds.labels[actual_idx]

            target = self.word_to_idx.get(word, 0) 

            return image, target

        except Exception as e:
            # Index-Fehler 
            return None