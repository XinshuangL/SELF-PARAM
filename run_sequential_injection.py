import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='mistralai/Mistral-7B-v0.1')
parser.add_argument('--dataset', type=str, default='PwC')
parser.add_argument('--num_samples_per_seq', type=int, default=20)
parser.add_argument('--num_seq', type=int, default=50)
args = parser.parse_args()

# Load data
from dsets import get_test_dataset
dataset = get_test_dataset(args.dataset, args.num_samples_per_seq*args.num_seq)

# Set context and QA pairs for test
context_list = []
test_qa_pairs = []
for context, questions, answers in dataset:
    context_list.append(context)
    test_qa_pairs.append({
        'questions': questions,
        'answers': answers,
    })

context_list_groups = [context_list[i:i+args.num_samples_per_seq] for i in range(0, len(context_list), args.num_samples_per_seq)]
test_qa_pairs_groups = [test_qa_pairs[i:i+args.num_samples_per_seq] for i in range(0, len(test_qa_pairs), args.num_samples_per_seq)]

results = []

for cur_context_list, cur_test_qa_pairs in zip(context_list_groups, test_qa_pairs_groups):
    # Load HuggingFace model
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Setup SELF-PARAM
    from self_param import SelfParam
    updatable_model = SelfParam(model, tokenizer)

    result = {}

    for i in range(1 + len(cur_context_list)):
        if i > 0:
            # Inject context
            cur_context = cur_context_list[i - 1]
            updatable_model.inject_context([cur_context], epochs=20)
            model = updatable_model.model

        # Test the model
        from utils import test_on_qa_pairs
        for j, cur_test_qa_pair in enumerate(cur_test_qa_pairs):
            f1, acc = test_on_qa_pairs(model, [cur_test_qa_pair], tokenizer)
            result[f'{i}-{j}'] = {
                'f1': f1,
                'acc': acc
            }

    results.append(result)

import json
with open('seq_results_tmp.json', 'w') as f:
    json.dump(results, f, indent=4)

mean_result = {}
for i in range(1 + args.num_samples_per_seq):
    for j in range(args.num_samples_per_seq):
        mean_result[f'{i}-{j}'] = {'f1': 0, 'acc': 0}

for result in results:
    for i in range(1 + args.num_samples_per_seq):
        for j in range(args.num_samples_per_seq):
            mean_result[f'{i}-{j}']['f1'] += (result[f'{i}-{j}']['f1'] / len(results))
            mean_result[f'{i}-{j}']['acc'] += (result[f'{i}-{j}']['acc'] / len(results))

# Save results
import json
with open(f'sequential_injection_result.json', 'w') as f:
    json.dump(mean_result, f, indent=4)
