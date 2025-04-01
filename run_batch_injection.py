import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='mistralai/Mistral-7B-v0.1')
parser.add_argument('--dataset', type=str, default='PwC')
parser.add_argument('--num_samples', type=int, default=100)
args = parser.parse_args()

# Load data
from dsets import get_test_dataset
dataset = get_test_dataset(args.dataset, args.num_samples)

# Set context and QA pairs for test
context_list = []
test_qa_pairs = []
for context, questions, answers in dataset:
    context_list.append(context)
    test_qa_pairs.append({
        'questions': questions,
        'answers': answers,
    })

# Load HuggingFace model
from transformers import AutoTokenizer, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(args.model)
tokenizer = AutoTokenizer.from_pretrained(args.model)

# Setup SELF-PARAM
from self_param import SelfParam
updatable_model = SelfParam(model, tokenizer)

# Inject context
updatable_model.inject_context(context_list)
model = updatable_model.model

# Test the model
from utils import test_on_qa_pairs
f1, acc = test_on_qa_pairs(model, test_qa_pairs, tokenizer)

# Save results
import json
with open(f'batch_injection_result.json', 'w') as f:
    json.dump({
        'f1': f1, 
        'acc': acc
    }, f, indent=4)
