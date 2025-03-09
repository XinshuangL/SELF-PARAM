# **Self-Updatable LLMs with Integrating Context into Model Parameters**

This is the official implementation of the paper [**Self-Updatable Large Language Models with Integrating Context into Model Parameters**](https://arxiv.org/abs/2410.00487), planned to be released by the end of March.



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
