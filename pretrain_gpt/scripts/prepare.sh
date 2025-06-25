#!/bin/bash
set -e

mkdir -p pretrain_gpt/data
python3 -c "from datasets import load_dataset; \
            ds = load_dataset('allenai/c4', 'en')['train']; \
            ds.to_json('pretrain_gpt/data/c4_corpus.json', lines=True)"

mkdir -p pretrain_gpt/tokenizer
curl -L -o pretrain_gpt/tokenizer/gpt2_merges.txt \
    "https://huggingface.co/openai-community/gpt2/resolve/main/merges.txt?download=true"
curl -L -o pretrain_gpt/tokenizer/gpt2_vocab.json \
    "https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json?download=true"

N_SPLITS=4
NPROC=$(nproc)
WORKERS=$((NPROC / N_SPLITS))

python3 Megatron-LM/tools/preprocess_data.py \
    --input pretrain_gpt/data/c4_corpus.json \
    --output-prefix pretrain_gpt/data/c4 \
    --vocab-file pretrain_gpt/tokenizer/gpt2_vocab.json \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file pretrain_gpt/tokenizer/gpt2_merges.txt \
    --json-keys text \
    --workers=$WORKERS \
    --partitions=$N_SPLITS \
    --append-eod

rm -rf .cache \
       pretrain_gpt/data/c4_corpus*.json