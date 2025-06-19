#!/bin/bash
set -e

python3 -c "from datasets import load_dataset; \
            ds = load_dataset('EleutherAI/pile', split='train', trust_remote_code=True); \
            ds.to_json("pile.json", lines=True)"

cd Megatron-LM
python3 tools/preprocess_data.py \
    --input ../pile-corpus.json \
    --output-prefix pile \
    --vocab-file ../pile-vocab.json \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file ../pile-merges.txt \
    --json-keys content \
    --workers $(nproc) \
    --append-eod