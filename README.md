# **Self-Updatable LLMs by Integrating Context into Model Parameters**

This is the official implementation of the paper [**Self-Updatable Large Language Models by Integrating Context into Model Parameters**](https://arxiv.org/abs/2410.00487).

### Environment
Please set OpenAI API key in the environment using `export OPENAI_API_KEY='your_api_key'`.

### How to Use the Method
Inject a piece of context (a list of text) into the HuggingFace model using the following script:
```python
# Prepare the model and context
model = AutoModelForCausalLM.from_pretrained("path_to_the_model")
tokenizer = AutoTokenizer.from_pretrained("path_to_the_model")
context_list = ['some context', 'some context']

# Setup SELF-PARAM
from self_param import SelfParam
updatable_model = SelfParam(model, tokenizer)

# Inject context
updatable_model.inject_context(context_list)
model = updatable_model.model
```

### Experiments on Question Answering
To reproduce the results of single context injection, please run:
```
python run_single_injection.py
```

To reproduce the results of batch context injection, please run:
```
python run_batch_injection.py
```

To reproduce the results of sequential context injection, please run:
```
python run_sequential_injection.py
```

### Experiments on Conversational Recommendation

We conduct the experiments on the following datasets and they are all included in this repo:
- inspired (`SELF-PARAM/LLMs-as-Zero-Shot-Conversational-RecSys/data/inspired`)
- redial (`SELF-PARAM/LLMs-as-Zero-Shot-Conversational-RecSys/data/redial`)

To reproduce the results, first run the following command to save out the model:
```
python main.py --config ... (todo)
```

With the models saved to `ckpt`, follow the commands below:
```
cd LLMs-as-Zero-Shot-Conversational-RecSys
sh train.sh
```
Then it will evaluate all models and output the results into four folders: 
1. Base Model (Mistral): `SELF-PARAM/LLMs-as-Zero-Shot-Conversational-RecSys/mistral`
2. SELF-PARAM on Mistral: `SELF-PARAM/LLMs-as-Zero-Shot-Conversational-RecSys/mistral-finetuned`
3. FT (Q): `SELF-PARAM/LLMs-as-Zero-Shot-Conversational-RecSys/mistral_context_instruct` 
4. FT (S): `SELF-PARAM/LLMs-as-Zero-Shot-Conversational-RecSys/mistral_qa_instruct`

We have attached the evaluation results in the corresponding folders `general/intermediate` under each folders shown above.

## Citations
If you find this repo helpful, please consider cite our paper:
```
@inproceedings{
    wang2025selfupdatable,
    title={Self-Updatable Large Language Models by Integrating Context into Model Parameters},
    author={Yu Wang and Xinshuang Liu and Xiusi Chen and Sean O'Brien and Junda Wu and Julian McAuley},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=aCPFCDL9QY}
}
```
