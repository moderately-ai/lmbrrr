#!/bin/sh
# Fetches the WikiText-2 raw test split (CC BY-SA, ~1.3 MB of text) into
# evals/ppl/wikitext2-test.txt for the quality reference battery
# (`lmbrrr ppl --text-file evals/ppl/wikitext2-test.txt`). Idempotent.
set -eu
cd "$(dirname "$0")"
if [ -s wikitext2-test.txt ]; then
    echo "evals/ppl/wikitext2-test.txt already present"
    exit 0
fi
# Same archive llama.cpp's get-wikitext-2.sh uses (the original Salesforce
# S3 link is dead), so our ppl corpus is byte-identical to theirs.
curl -fsSL -o wikitext-2-raw-v1.zip \
    https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
unzip -p wikitext-2-raw-v1.zip wikitext-2-raw/wiki.test.raw > wikitext2-test.txt
rm wikitext-2-raw-v1.zip
wc -c wikitext2-test.txt
