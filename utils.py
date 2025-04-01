import random
import torch
from torch.optim import AdamW
from torch.nn import functional as F
from peft import get_peft_model, LoraConfig, TaskType
from chatgpt import gpt_generate
from metrics import qa_f1_score, simple_accuracy
import numpy as np
from tqdm import tqdm

def text2id(tokenizer, text, max_len=100000, add_special_tokens=False):
    ids = tokenizer(text, add_special_tokens=add_special_tokens, truncation=True, max_length=max_len).input_ids
    ids = torch.tensor(ids).view(1, -1).long()
    return ids

def get_optimizer(model, use_lora, lr=2e-5):
    if use_lora:
        lora_config = {
            "inference_mode": False,
            "r": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": ["q_proj", "v_proj", "k_proj", 'up_proj', 'down_proj', 'gate_proj']
        }
        
        peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM, 
                    inference_mode=lora_config['inference_mode'], 
                    r=lora_config['r'], 
                    lora_alpha=lora_config['lora_alpha'], 
                    lora_dropout=lora_config['lora_dropout'],
                    target_modules=lora_config.get('target_modules', None)
                )
        model = get_peft_model(model, peft_config)
        params_to_optimize = [param for param in model.parameters() if param.requires_grad]
        optimizer = AdamW(params_to_optimize, lr=lr)
    else:
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if 'mlp' in name:  # Adjust this condition based on your model's architecture
                param.requires_grad = True
        params_to_optimize = [param for param in model.parameters() if param.requires_grad]
        optimizer = AdamW(params_to_optimize, lr=lr)
    return optimizer, model

def distillation_loss(s_out, t_out, T=3):
    loss = F.kl_div(F.log_softmax(s_out / T, dim=-1), F.log_softmax(t_out / T, dim=-1), log_target=True) * (T**2)
    return loss

def train_with_context(model, optimizer, context, base_model, tokenizer, gt_model):
    model.train()
    s_ids = torch.tensor([tokenizer.bos_token_id]).unsqueeze(0)
    context_ids = text2id(tokenizer, context)
    random_index = random.randint(1, context_ids.shape[1] - 2)
    context_ids, finetune_ids = context_ids[:, :random_index], context_ids[:, random_index:]

    base_input = torch.cat([
        s_ids,
        context_ids,
        finetune_ids
    ], dim=1)

    finetune_input = torch.cat([
        s_ids,
        finetune_ids
    ], dim=1)

    with torch.no_grad():
        if gt_model == 'finetune':
            base_logits = model(base_input.cuda(0)).logits.detach()
            selected_base_logits = base_logits[:, s_ids.shape[1] + context_ids.shape[1] - 1: base_input.shape[1] - 1]
        elif gt_model == 'base':
            base_logits = base_model(base_input.cuda(1)).logits
            selected_base_logits = base_logits[:, s_ids.shape[1] + context_ids.shape[1] - 1: base_input.shape[1] - 1]
        elif gt_model == 'none':
            base_logits = base_model(finetune_input.cuda(1)).logits
            selected_base_logits = base_logits[:, s_ids.shape[1] - 1: finetune_input.shape[1] - 1]
        else:
            raise NotImplementedError

    finetune_logits = model(finetune_input.cuda(0)).logits
    selected_finetune_logits = finetune_logits[:, s_ids.shape[1] - 1: finetune_input.shape[1] - 1]

    loss = distillation_loss(selected_finetune_logits.cuda(0), selected_base_logits.cuda(0), T=1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train_with_qa_pair(model, optimizer, context, qa_pair, base_model, tokenizer, gt_model):
    model.train()

    s_ids = torch.tensor([tokenizer.bos_token_id]).unsqueeze(0)
    context_ids = text2id(tokenizer, context)
    finetune_ids = text2id(tokenizer, qa_pair)

    base_input = torch.cat([
        s_ids,
        context_ids,
        finetune_ids
    ], dim=1)

    finetune_input = torch.cat([
        s_ids,
        finetune_ids
    ], dim=1)

    with torch.no_grad():
        if gt_model == 'finetune':
            base_logits = model(base_input.cuda(0)).logits.detach()
            selected_base_logits = base_logits[:, s_ids.shape[1] + context_ids.shape[1] - 1: base_input.shape[1] - 1]
        elif gt_model == 'base':
            base_logits = base_model(base_input.cuda(1)).logits
            selected_base_logits = base_logits[:, s_ids.shape[1] + context_ids.shape[1] - 1: base_input.shape[1] - 1]
        elif gt_model == 'none':
            base_logits = base_model(finetune_input.cuda(1)).logits
            selected_base_logits = base_logits[:, s_ids.shape[1] - 1: finetune_input.shape[1] - 1]
        else:
            raise NotImplementedError

    finetune_logits = model(finetune_input.cuda(0)).logits
    selected_finetune_logits = finetune_logits[:, s_ids.shape[1] - 1: finetune_input.shape[1] - 1]

    loss = distillation_loss(selected_finetune_logits.cuda(0), selected_base_logits.cuda(0), T=1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def generate_qa_pairs(context_list, generate_num, generation_model='gpt-4o-mini'):
    cur_data = []
    for context in context_list:
        while_times = 0
        qa_pairs = []
        while len(qa_pairs) < generate_num:
            while_times += 1
            if while_times > 500:
                print('while_times', while_times)
                break
            decoded_sentence = gpt_generate(context, gpt_model=generation_model)

            for line in decoded_sentence.split("\n"):
                qas = line.split("Question:")
                for qa in qas:
                    qa = qa.strip()
                    if len(qa) > 5 and 'Answer: ' in qa:
                        if len(qa.split('Answer: ')[-1]) >= 1:
                            if len(qa.split('Answer: ')[0]) >= 1:
                                qa_pairs.append("Question: " + qa)

        qa_pairs = qa_pairs[: generate_num]
        cur_data.append(qa_pairs)
    return cur_data

def test_on_one_qa_pair(questions, answers, model, tokenizer, max_input_tokens, max_new_tokens):
    model.eval()
    f1 = 0
    acc = 0
    count = 0
    for question, answer in zip(questions, answers):
        prefix = torch.tensor([tokenizer.bos_token_id]).unsqueeze(0).to(model.device)
        question_ids = text2id(tokenizer, question).to(model.device)
        if question_ids.shape[1] > max_input_tokens - prefix.shape[1]:
            question_ids = question_ids[:, question_ids.shape[1] - (max_input_tokens - prefix.shape[1]):]
        question_ids = torch.cat([prefix, question_ids], dim=1)
        pred_answer_ids = model.generate(input_ids=question_ids, 
                                    max_new_tokens=max_new_tokens,
                                    pad_token_id=tokenizer.eos_token_id)
        pred_answer_ids = pred_answer_ids.detach().cpu().numpy()[0][question_ids.shape[1]:]
        pred_answer = tokenizer.decode(pred_answer_ids, skip_special_tokens=True)
        f1 += qa_f1_score(pred_answer.replace("</s>", "").strip(), answer.replace("</s>", "").strip())
        acc += simple_accuracy(pred_answer, answer)
        count += 1
    return f1 / count, acc / count

def test_on_qa_pairs(model, test_qa_pairs, tokenizer, max_input_tokens=2000, max_new_tokens=10):
    accs = []
    f1s = []
    for data in tqdm(test_qa_pairs):
        questions = data['questions']
        answers = data['answers']
        f1, acc = test_on_one_qa_pair(questions, answers, model, tokenizer, max_input_tokens, max_new_tokens)
        f1s.append(f1)
        accs.append(acc)
    return np.mean(f1s), np.mean(accs)
