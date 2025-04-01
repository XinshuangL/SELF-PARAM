import os
from transformers import LlamaTokenizer
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import json
from datasets import load_dataset
import itertools


class UnifiedDataset(Dataset):
    def __init__(self, dataset, dataset_name, num):
        self.dataset = dataset
        print('original dataset len: ', len(dataset))
        self.dataset_name = dataset_name
        self.num = min(num, len(self.dataset)) if num > 0 else len(self.dataset)
        self.exit = iter([])
        assert dataset_name in ['PwC', 'inspired', 'redial', 'Chat']

    def __getitem__(self, index):
        if index < self.num:
            data = self.dataset.__getitem__(index)
            if self.dataset_name == 'PwC':
                return [data[0]], data[1], data[2]
            elif self.dataset_name == 'inspired' or self.dataset_name == 'redial':
                return data[0], data[1], data[2]
            else:
                raise NotImplementedError
        else:
            return next(self.exit)

    def __len__(self):
        return self.num

class ChatDataset(Dataset):
    def __init__(self):
        chat_filename = 'data/memory_bank_en.json'
        with open(chat_filename, 'r') as f:
            self.conversations = json.load(f)
        f.close()

        question_filename = 'data/probing_questions_en.jsonl'

        self.questions = []
        with open(question_filename) as f:
            for line in f:
                json_data = json.loads(line.strip())
                self.questions.append(json_data)
        f.close()

        # check the orders
        conversation_keys = [
            self.conversations[x]['name']
            for x in self.conversations
        ]
        for idx, question in enumerate(self.questions):
            assert list(question.keys())[0] == conversation_keys[idx]

        self.conversations = [
            self.conversations[x]['history']
            for x in self.conversations
        ]
        self.questions = [
            list(question.values())[0]
            for question in self.questions
        ]

    def __getitem__(self, index):
        return self.conversations[index], self.questions[index]

    def __len__(self):
        return len(self.conversations)

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

class PwCDataset(Dataset):
    def __init__(self):
        dataset = load_dataset("sggetao/PwC", split='test')
        train_dataset = load_dataset("sggetao/PwC", split='train')
        pre_context = ''
        self.context = []
        self.questions = []
        self.answers = []

        tok = LlamaTokenizer.from_pretrained('openlm-research/open_llama_3b_v2')
        for data in itertools.chain(dataset, train_dataset):
            context = data['input']
            question = data['prompt']
            answer = data['answer']
            context = context.replace("<s>", "")
            question = question.replace("<s>", "")
            answer = answer.replace("<s>", "")
            if len(tok(answer, add_special_tokens=False)['input_ids']) > 5:
                continue
            if 'above text' in question.lower():
                continue

            if not context == pre_context:
                self.context.append('Context: ' + context)
                self.questions.append([])
                self.answers.append([])
            pre_context = context
            self.questions[-1].append('Question: ' + question + ' Answer:')
            self.answers[-1].append(answer)

    def __getitem__(self, index):
        return self.context[index], self.questions[index], self.answers[index]

    def __len__(self):
        return len(self.context)

def get_test_dataset(dataset_name, num):
    if dataset_name == 'PwC':
        test_dataset = PwCDataset()

    elif dataset_name == 'Chat':
        test_dataset = ChatDataset()

    elif dataset_name == 'inspired' or dataset_name == 'redial':
        test_dataset = ConversationalRecommendationDataset(dataset_name)

    else:
        raise NotImplementedError

    test_dataset = UnifiedDataset(test_dataset, dataset_name, num)
    return test_dataset
