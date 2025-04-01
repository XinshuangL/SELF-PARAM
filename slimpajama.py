import torch
import random
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer

def split_sequence(seq_length, min_length, max_length):

    if seq_length < min_length:
        return [seq_length]
    
    # Initialize the chunks
    chunks = []
    remaining_length = seq_length

    while remaining_length > 0:
        # Calculate the maximum length for the current chunk
        max_chunk_length = min(remaining_length, max_length)
        
        # Ensure the chunk is at least min_length characters long and handle the remaining length
        if remaining_length <= max_length:
            if remaining_length < min_length and chunks:
                # Adjust previous chunks to make the last chunk at least min_length
                needed = min_length - remaining_length
                for i in range(len(chunks) - 1, -1, -1):
                    if chunks[i] - needed >= min_length:
                        chunks[i] -= needed
                        remaining_length += needed
                        break
                    else:
                        needed -= (chunks[i] - min_length)
                        remaining_length += (chunks[i] - min_length)
                        chunks[i] = min_length

            chunk_length = remaining_length if remaining_length >= min_length else min_length

        else:
            chunk_length = random.randint(min_length, max_chunk_length)
        
        # Append the chunk length to the list
        chunks.append(chunk_length)
        
        # Reduce the remaining length
        remaining_length -= chunk_length

    return chunks


class SlimPajamaDataset(Dataset):
    def __init__(self, root='DKYoon/SlimPajama-6B', 
                 split='train', 
                 tokenizer=None,
                 tokenizer_path=None,
                 min_length=256,
                 max_length=768,
                 num_tokens=768,
                 max_seq_length=None,
                 end_special_token="",
                 add_special_tokens=False,
                 target_is_context=False,
                 shuffle_first_context=False,
                 overlap_contexts=False,
                 negative_contexts_ratio=0.0,
                 overlap_contexts_ratio=0.0,
                 target_is_context_ratio=0.0,
                 repeat_with_unrelated=False,
                 max_unrelated_at_one_step=20,
                 force_num_of_contexts=None,
                 ):
        self.root = root
        self.max_length = max_length
        self.min_length = min_length
        self.num_tokens = num_tokens
        self.end_special_token = end_special_token
        self.add_special_tokens = add_special_tokens
        self.max_seq_length = max_seq_length if max_seq_length is not None else max_length
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.target_is_context = target_is_context
        self.shuffle_first_context = shuffle_first_context
        self.overlap_contexts = overlap_contexts
        self.negative_contexts_ratio = negative_contexts_ratio
        self.overlap_contexts_ratio = overlap_contexts_ratio
        self.target_is_context_ratio = target_is_context_ratio
        # self.instruction = instruction
        self.last_contexts = None
        self.repeat_with_unrelated = repeat_with_unrelated
        self.max_unrelated_at_one_step = max_unrelated_at_one_step
        self.force_num_of_contexts = force_num_of_contexts
        if self.repeat_with_unrelated:
            self.unrelated_contexts = []

        assert not (self.target_is_context_ratio > 0 and self.target_is_context), "you cannot set target_is_context_ratio > 0 and target_is_context=True at the same time!"

        self.ds = load_dataset(root)[split]

    def get_context_and_sentence(self, doc):
        doc = self.tokenizer(doc + self.end_special_token, return_tensors='pt', truncation=False, add_special_tokens=False).input_ids[0]
        return doc[:self.max_length]

    def get_repeat_context_sentence(self, doc):

        if self.repeat_with_unrelated:

            contexts, sentence = self.get_context_and_sentence(doc, return_doc=True)

            # randomly pick some contexts from self.unrelated_contexts:
            num_unrelated = min(self.max_unrelated_at_one_step, len(self.unrelated_contexts))

            np.random.shuffle(self.unrelated_contexts)
            unrelated_contexts = self.unrelated_contexts[:num_unrelated]
            contexts.extend(unrelated_contexts)
            
            # save the context to construct the set of unrelated_contexts
            if self.repeat_with_unrelated:
                if len(self.unrelated_contexts) < 200:
                    self.unrelated_contexts.append(contexts[0])
                else:
                    self.unrelated_contexts.pop(0)
                    self.unrelated_contexts.append(contexts[0])

            return contexts, sentence, 3
        
        else:

            output = self.get_context_and_sentence(doc)
            if output is None:
                return None
            contexts, sentence, _ = output
            # sentence = torch.cat(contexts)
            
            return contexts, sentence, 4
        
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        data = self.ds[idx]
        doc = data['text']
        output = self.get_context_and_sentence(doc)
        return output

