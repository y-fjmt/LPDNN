#!/bin/bash
set -e

python3 -c "from datasets import load_dataset; \
            ds = load_dataset('allenai/c4', 'en'); \
            ds.to_json("c4.json", lines=True)"

cd Megatron-LM
python3 tools/preprocess_data.py \
    --input ../c4-corpus.json \
    --output-prefix c4 \
    --vocab-file ../c4-vocab.json \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file ../c4-merges.txt \
    --json-keys content \
    --workers $(nproc) \
    --append-eod