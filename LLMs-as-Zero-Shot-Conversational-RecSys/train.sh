# Original Model without fine-tuning and without in-context examples
python localmodels.py --config mistral/general/inspired_config.yaml
python tools/post_fix.py --from_json mistral/general/inspired_test.jsonl \
    --prompt_config mistral/general/inspired_config.yaml
python mistral/general/extract.py --dataset inspired
python tools/evaluate.py \
    --from_json mistral/general/intermediate/inspired/extracted.jsonl

python localmodels.py --config mistral/general/redial_config.yaml
python tools/post_fix.py --from_json mistral/general/redial_test.jsonl \
    --prompt_config mistral/general/redial_config.yaml
python mistral/general/extract.py --dataset redial
python tools/evaluate.py \
    --from_json mistral/general/intermediate/redial/extracted.jsonl

# SELF-PARAM
python localmodels.py --config mistral-finetuned/general/inspired_config.yaml
python tools/post_fix.py --from_json mistral-finetuned/general/inspired_test.jsonl \
    --prompt_config mistral-finetuned/general/inspired_config.yaml
python mistral-finetuned/general/extract.py --dataset inspired
python tools/evaluate.py \
    --from_json mistral-finetuned/general/intermediate/inspired/extracted.jsonl

python localmodels.py --config mistral-finetuned/general/redial_config.yaml
python tools/post_fix.py --from_json mistral-finetuned/general/redial_test.jsonl \
    --prompt_config mistral-finetuned/general/redial_config.yaml
python mistral-finetuned/general/extract.py --dataset redial
python tools/evaluate.py \
    --from_json mistral-finetuned/general/intermediate/redial/extracted.jsonl

# FT (S)
python localmodels.py --config mistral_qa_instruct/general/inspired_config.yaml
python tools/post_fix.py --from_json mistral_qa_instruct/general/inspired_test.jsonl \
    --prompt_config mistral_qa_instruct/general/inspired_config.yaml
python mistral_qa_instruct/general/extract.py --dataset inspired
python tools/evaluate.py \
    --from_json mistral_qa_instruct/general/intermediate/inspired/extracted.jsonl

python localmodels.py --config mistral_qa_instruct/general/redial_config.yaml
python tools/post_fix.py --from_json mistral_qa_instruct/general/redial_test.jsonl \
    --prompt_config mistral_qa_instruct/general/redial_config.yaml
python mistral_qa_instruct/general/extract.py --dataset redial
python tools/evaluate.py \
    --from_json mistral_qa_instruct/general/intermediate/redial/extracted.jsonl

# FT (C)
python localmodels.py --config mistral_context_instruct/general/inspired_config.yaml
python tools/post_fix.py --from_json mistral_context_instruct/general/inspired_test.jsonl \
    --prompt_config mistral_context_instruct/general/inspired_config.yaml
python mistral_context_instruct/general/extract.py --dataset inspired
python tools/evaluate.py \
    --from_json mistral_context_instruct/general/intermediate/inspired/extracted.jsonl

python localmodels.py --config mistral_context_instruct/general/redial_config.yaml
python tools/post_fix.py --from_json mistral_context_instruct/general/redial_test.jsonl \
    --prompt_config mistral_context_instruct/general/redial_config.yaml
python mistral_context_instruct/general/extract.py --dataset redial
python tools/evaluate.py \
    --from_json mistral_context_instruct/general/intermediate/redial/extracted.jsonl