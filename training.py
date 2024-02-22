import pandas as pd
import torch
from torch.utils.data.dataloader import Dataset, DataLoader
import numpy as np
from LSTM import LSTM
from VideoToPose import landmark_from_json
from CustomOneHot import OneHot
from time import time



class VideoLookupDataset(Dataset):
    
    def __init__(self, dataframe, encoder, sequence_length):
        self.dataframe = dataframe
        self.encoder = encoder
        
    def __getitem__(self, index):
        
        row = self.dataframe.iloc[index].to_numpy()
        features = landmark_from_json(row[2])['hands_only']
        label = self.encoder.encode(row[0])
        
        # Padding deatures
        pad_value = sequence_length - np.size(features, 0)
        features = np.pad(features, ((0, pad_value), (0, 0)), mode='constant', constant_values=(0))
        
        
        
        features, label = torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
        
        features.to(0)
        label.to(0)
        
        return features, label
    
    def __len__(self):
        return len(self.dataframe)

'''
If dataset changes, only the file locations, and the lookup dataframe needs to be changed.
The splitting and training will remain the same!
'''

def train(num_epoch, dataloader, network, loss_func, optimiser, label_encoder):
    for epoch in range(num_epoch):
        correct = 0
        total = 0
        
        start_time = time()
        for batch, (features, label) in enumerate(dataloader):
            duration = time() - start_time
            
            label = torch.squeeze(label)
            
            optimiser.zero_grad()
            logits = network(features)
            
            
            loss = loss_func(logits, label)
            loss.backward()
            optimiser.step()
            
            if torch.argmax(logits) == torch.argmax(label):
                correct += 1
            total += 1
            
            
            
            
            print(f"    {batch + 1}/{int(len(dataloader))} batch, {loss} loss, {duration} seconds")
            start_time = time()
        print(f"{epoch + 1}/{num_epoch} epoch, {(correct / total) * 100} accuracy")
            
    
def checkpoint(state, filename="default_checpoint.pth.tar"):
    torch.save(state, filename)

        
    
    
if __name__ == "__main__":
    
    lookup_df = pd.read_csv("dataset/wlasl_lookup.csv", converters={'video_id': str})
    
    glossary = np.unique(lookup_df['gloss'].values)    
    
    # Hyperparameters
    hand_input_size = 2 * 21 * 2
    hand_hidden_size = hand_input_size * 3
    face_input_size = 468 # I am looking to reduce this number to only include the eyes, brows and mouth
    epoch = 10
    layers = 4
    learning_rate = 0.0005
    num_classes = len(glossary)
    sequence_length = 233 # from External script that runs through the json files (took 233 seconds)
    
    
    onehot_encoder = OneHot()
    onehot_encoder.fit_categories(glossary)
    
    train_df = lookup_df.loc[lookup_df['gloss'] == 'cool']
    train_dataset = VideoLookupDataset(train_df, onehot_encoder, sequence_length)
    train_dataloader = DataLoader(train_dataset, batch_size=2)
        
    # Defining model
    slr_model = LSTM(hand_input_size, hand_hidden_size, num_classes, layers)
    optimiser = torch.optim.Adam(slr_model.parameters(), lr=learning_rate, weight_decay=0.005)
    loss_func = torch.nn.CrossEntropyLoss()   

        
    train(
        num_epoch=epoch,
        dataloader=train_dataloader,
        network=slr_model,
        loss_func=loss_func,
        optimiser=optimiser,
        label_encoder=onehot_encoder
        )
    
    save_checkpoint = {
        'state_dict': slr_model.state_dict,
        'optimiser': optimiser.state_dict
    }
    
    checkpoint(save_checkpoint)