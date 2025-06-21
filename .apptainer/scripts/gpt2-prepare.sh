#!/bin/bash
set -e

python3 -c "from datasets import load_dataset; \
            ds = load_dataset('allenai/c4', 'en')['train']; \
            ds.to_json('c4_corpus.json', lines=True)"

cd Megatron-LM

curl -L -o merges.txt \
    "https://huggingface.co/openai-community/gpt2/resolve/main/merges.txt?download=true"
curl -L -o gpt2_vocab.json \
    "https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json?download=true"

python3 tools/preprocess_data.py \
    --input ../c4_corpus.json \
    --output-prefix c4 \
    --vocab-file ../gpt2_vocab.json \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file ../gpt2_merges.txt \
    --json-keys content \
    --workers $(nproc) \
    --append-eod