# If You Give a Policy a Bit

If you give a policy a bit, it's going to want to learn a secret.

If it wants to learn a secret, it's going to need log₂(N) bits of information.

If it only gets 1 bit per episode... it's going to take O(N) episodes.


<figure>
  <img width="308" height="316" alt="Screenshot 2026-01-05 at 11 29 20 PM" src="https://github.com/user-attachments/assets/067289b9-d0b9-4dea-8957-f093cd63756b" />

  
  <figcaption>A few early tests of the the RL binary secret bandit</figcaption>
</figure>

---

## What is this?

This is an empirical test of a claim from the [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) post: **scalar rewards bottleneck RL to ~1 bit of learning per episode**. The argument is information-theoretic. If you only tell the model "good" or "bad" after each episode, that's 1 bit of feedback. So it should take about N episodes to learn log₂(N) bits of information.

Is this actually true? I don't know. [Some people think yes](https://x.com/_arohan_/status/1973260831824683173). [Some people think no](https://x.com/khoomeik/status/1979236175820001707). So I built the simplest possible test.

## The test

Pick a random integer S ∈ [0, N) and keep it fixed for the entire training run. The model's job is to memorize S. Since S is uniformly random, learning it requires exactly log₂(N) bits. There's no pattern to exploit, no shortcut. The model has to absorb the information from the reward signal.

Then vary the reward signal:

| Reward Type | What it tells the model | Bits per episode |
|-------------|------------------------|------------------|
| **Binary** | correct or not | 1 |
| **Binned** | how close (discretized) | log₂(B) |
| **Dense** | per-bit feedback over k turns | log₂(N) |

Binary reward is the "straw" - minimal bandwidth. Dense reward is the "firehose" - maximal bandwidth. If the bottleneck theory is right, the straw should be much slower than the firehose.

## Measuring learning

I can't just wait for the model to guess correctly. That could take forever, and I wouldn't know if it was learning anything in the meantime.

Instead I measure how much the model knows at any point using logprobs. Give the model the correct answer and ask: how surprised is it? If it assigns probability 1/N to the secret (uniform), it knows nothing. If it assigns probability 1, it knows everything.

Convert this to bits:

```
bits_known = log₂(N) + log P(S) / ln(2)
```

When `bits_known ≈ 0`, the model is guessing. When `bits_known ≈ 10`, the model has memorized a 10-bit secret.

## Running experiments

```bash
pip install tinker
export TINKER_API_KEY=sk-...

# Binary reward (1 bit/ep)
python train.py env_type=single_step N=64 reward_type=binary

# Binned reward (log₂(B) bits/ep)
python train.py env_type=single_step N=64 reward_type=binned_log_distance reward_bins=8

# Dense reward (log₂(N) bits/ep)
python train.py env_type=multi_step N=64

# Run all conditions
python run_sweep.py --sequential
```

## Things I learned

**Format learning is a real problem.** If the model can't reliably output integers in the right format, RL will spend its limited feedback teaching syntax instead of the secret. I fix this with a warmup phase that trains on uniform random numbers before the actual experiment.

**B=2 binned is not the same as binary.** With log-distance binning, B=2 ends up being very forgiving. At N=1024, any guess within ~30 of the secret maps to the "good" bin. For fair 1-bit baselines, use true binary reward.

**You need group_size ≥ 2.** With GRPO-style training (no value network), a single trajectory gives zero advantage signal. The baseline is just the reward itself, so the advantage is always zero. Nothing learns.

## Checkpoints

I provide warmup checkpoints so you can skip format training:

```python
SINGLE_STEP_CHECKPOINT = "tinker://61ffdf2c-c9ae-52f1-8b0a-d5757c68bee8:train:0/weights/final"
MULTI_STEP_CHECKPOINT = "tinker://1e79325e-97ad-5cfc-aae3-fdc7b5951746:train:0/weights/final"
```

## References

- [Thinking Machines - LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)
- [Ord (2025) - The Inefficiency of RL](https://www.tobyord.com/writing/inefficiency-of-reinforcement-learning)
- [Li - Information Bandwidth of RL](https://richardli.xyz/post/information-bandwidth-rl/)

---

Part of [Thinking Machines Lab](https://thinkingmachines.ai/)'s community projects.
