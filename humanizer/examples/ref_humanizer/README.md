# Reference humanizer (example submission)

A complete, minimal submission for the SN32 humanizer competition: Qwen3-4B-Instruct-2507 fine-tuned with LoRA
to turn AI-written text back into human-written text. It is the owner's format example, not a strong
humanizer: copy the layout, replace the model.

## What a submission is

A bundle (a `.tar` file) evaluated on one H100 (80 GB) with no network, within 60 minutes, at most 50 GB:

```
submission.json     {"name", "entrypoint": "module:function", "license", "batch_size"}
src/                your code; the entrypoint module lives here
requirements.txt    what the harness installs from wheels/ (offline)
wheels/             every Python dependency as a wheel, for the harness image's Python
weights/            model files
```

* `license` must be `MIT`, `Apache-2.0`, `BSD-3-Clause` or `OWNER-GRANT`.
* The entrypoint receives batches of `batch_size` items `{"text", "lang", "genre"}` and returns one string per
  item, in order: `humanize(items: list[dict]) -> list[str]`.
* The harness image is `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime` (torch 2.7.1 is preinstalled; build
  your wheels with that image's pip, as below, so they match its Python). Your code
  runs as an unprivileged user on a read-only filesystem (writable: `/tmp`), with the bundle mounted at
  `/sub` (`HUMEVAL_SUB`). `random`, `numpy` and `torch` are seeded from `HUMEVAL_SEED` before the first call.
* Outputs longer than 1.3 × the input's words are cut; an empty output counts as a failed output.

## How outputs are scored

Every output goes through quality gates (meaning kept, same language and genre, natural text) that give it a
level of 1, 0.5 or 0, then through the crown detectors (GPTZero, ZeroGPT, Pangram). The score combines how far
the detectors are from telling your outputs apart from human texts on the same topics and how human they
rate your outputs, weighted by the quality level. See the competition spec for the exact metric.

## This example

1. **Pairs** (`scripts/ref_humanizer_v2_pairs.py` in the owner repo): human texts of the production genre mix
   from the pool are rewritten "as an AI assistant would write it" by the same LLMs that write the period
   inputs. The model learns the reverse direction: AI text -> the original human text.
2. **Training** (`train/train_lora.py`): LoRA r = 16 on all linear layers, 2 epochs, merged into bf16 weights.

   ```
   pip install transformers==4.57.1 peft==0.17.1
   python train/train_lora.py --pairs pairs.jsonl --out bundle/
   ```
3. **Bundle**: copy `src/`, `submission.json`, `requirements.txt` into `bundle/`, download the wheels with the
   harness image's pip, and tar it:

   ```
   docker run --rm --entrypoint pip -v $PWD/bundle/wheels:/w -v $PWD/requirements.txt:/r.txt:ro \
       pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime download -r /r.txt -d /w
   tar -cf bundle.tar -C bundle --exclude=./adapter --exclude=./train_log.json .
   ```
4. **Local check**: run `harness/run.py` from the owner repo with `HUMEVAL_SUB=bundle HUMEVAL_IN=<dir with
   items.jsonl> HUMEVAL_OUT=out` to see exactly what the evaluator will get.
5. **Submit**: upload `bundle.tar` to your Hugging Face repo, commit `<repo>@<commit sha>` on chain and
   submit, all in one command: `scripts/miner_submit.py ... publish --repo <you>/<repo> --file bundle.tar`.
   The full guide is [`docs/miners.md`](../../docs/miners.md).
