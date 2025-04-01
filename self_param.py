import copy
from slimpajama import SlimPajamaDataset
from utils import get_optimizer, train_with_qa_pair, train_with_context, generate_qa_pairs
from tqdm import tqdm
import random
import threading
import time

class SelfParam:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        SlimPajama_dataset = SlimPajamaDataset(tokenizer=self.tokenizer, target_is_context=True, max_length=256)
        self.pretrained_data_iter = iter(SlimPajama_dataset)

    def generate_QA_pairs(self, texts, generate_num, generation_model):
        data_list = []
        for text in texts:
            data_list.append({'context': text})

        def add_qa_pairs_to_dict(idx, data_list):
            data_list[idx]['qa_pairs'] = generate_qa_pairs(data_list[idx]['context'], generate_num, generation_model)

        threads = []
        for idx in tqdm(range(len(data_list))):
            execute_thread = threading.Thread(target=add_qa_pairs_to_dict, args=(idx, data_list))
            threads.append(execute_thread)
            execute_thread.start()
            
            if len(threads) == 100:
                for execute_thread in threads:
                    execute_thread.join()
                threads = []
                time.sleep(10)

        for execute_thread in threads:
            execute_thread.join()

        return data_list

    def inject_context(self, texts, use_lora=True, epochs=50, num_per_text=10, use_train_with_context=True, use_context_for_general_text=True, generation_model='gpt-4o-mini'):
        # Prepare Models
        base_model = copy.deepcopy(self.model.cpu()).cuda(1)
        self.model.cuda(0)        
        optimizer, self.model = get_optimizer(self.model, use_lora)

        # Prepare Data
        data_list = self.generate_QA_pairs(texts, epochs * num_per_text, generation_model)

        # Inject context
        for epoch in tqdm(range(1, epochs + 1)):
            merged_data_list = []
            for data in data_list:
                for context, qa_pairs in zip(data['context'], data['qa_pairs']):       
                    cur_qa_pairs = qa_pairs[(epoch-1)*num_per_text:epoch*num_per_text]
                for qa_pair in cur_qa_pairs:
                    merged_data_list.append({
                        'context': context,
                        'qa_pair': qa_pair
                    })
            random.shuffle(merged_data_list)

            for data_dict in merged_data_list:
                context = data_dict['context']
                qa_pair = data_dict['qa_pair']
                train_with_qa_pair(self.model, optimizer, context, qa_pair, base_model, self.tokenizer, 'base')

                if use_train_with_context:
                    train_with_context(self.model, optimizer, context, base_model, self.tokenizer, 'base')

                RedPajama_ids = next(self.pretrained_data_iter)
                RedPajama_text = self.tokenizer.decode(RedPajama_ids)
                if use_context_for_general_text:
                    train_with_qa_pair(self.model, optimizer, context, RedPajama_text, base_model, self.tokenizer, 'base')
                else:
                    train_with_qa_pair(self.model, optimizer, context, RedPajama_text, base_model, self.tokenizer, 'none')
                
