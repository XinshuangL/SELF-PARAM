import os
import json

class ConversationalRecommendationDataset(Dataset):
    def __init__(self, dataset, train=True):
        if train:
            filename = f'data/{dataset}/conversations.json'
            conversation_labels = f"data/{dataset}/conversation_labels.json"
        else:
            raise NotImplementedError

        with open(filename, 'r') as f:
            self.data = json.load(f)

        print("Raw data length:", len(self.data))
        if os.path.exists(conversation_labels):
            with open(conversation_labels, 'r') as f:
                self.labels = json.load(f)
            indices = [int(i) for i, x in self.labels.items() if x != '0']
            self.data = [self.data[i] for i in indices]
            print("After filtering with conversation labels, data length:", len(self.data))
        else:
            print("Warning: conversation labels not found") 
            print("Using all data")
            
    def __getitem__(self, index):

        conversation_turns = self.data[index]
        prompt = ''
        last_turn = 'user'
        for turn in conversation_turns[1:]:
            if last_turn == 'user':
                prompt += "System: " + turn + '\n'
                last_turn = 'system'
            else:
                prompt += "User: " + turn + '\n'
                last_turn = 'user'
        return [prompt.strip()], "", ""

    def __len__(self):
        return len(self.data)