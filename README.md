# Enhancing LLM Security Against Indirect Prompt Injection via Direct Preference Optimization

Ruoyao Wen, Xinhang Ma

## Instructions

Before we start, unzip ```data.7z``` under the repo.

We recommend using the [official docker image for LLaMA-Factory](https://hub.docker.com/r/hiyouga/llamafactory) to most easily get started. Inside the container (copy this folder or mount it), then you can directly run the following script:

```bash
run.sh
```

The only change you need to make is to set your desired `cache_dir` and use you own Hugging Face token (`HUGGING_FACE_HUB_TOKEN`). Plus `OUT_DIR` if you wish, by default it is set to `out`, we recommend changing it to something more descriptive for the experiments.

The script defaults to fine-tune with DPO Llama-3-8B model on 200 samples from our multi-turn preference dataset using LoRA with all hyperparameters reported in our paper. After fine-tuning, it evaluates the model on 500 samples that are unseen during training for ASR. 

To evaluate for utility, you need to install the [alpaca-eval](https://github.com/tatsu-lab/alpaca_eval) pacakge and set `--reference_outputs data/davinci_003_outputs.json`. 
By default, `--model_outputs` should be set to, for example `saves/Meta-Llama-3-8B-Instruct/lora/$OUT_DIR-log/predictions_on_davinci_003_outputs.json`, after running the script above.
The results presented in our paper uses `gpt-5-mini` as the evaluator model with all other settings defalting to those in the alpaca-eval repository.


## Notes

All of our experiments were run on a single NVIDIA H100 GPU. Depending on your hardware, you may increase/decrease the `per_device_train_batch_size` (and therefore decrease/increase the `gradient_accumulation_steps`) during training to maintain an effective batch size of `64`. Similarly, you may adjust the `batch_size` during evaluation.

# Final Report


[Final report](DRL_project_report.pdf) is uploaded to the repo.
