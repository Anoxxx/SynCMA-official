# A Synchronous Information Geometric Method for High-Dimensional Online Optimization
This is the official repository of paper 'An Invariant Information Geometric Method for High-Dimensional Online Optimization, consisting the source-code and full version of the work.

Except for the common modules (e.g., numpy, scipy), our source code depends on the following modules.

- Mandatory
  - PyPop7 (https://github.com/Evolutionary-Intelligence/pypop)
  - mujoco-py (https://github.com/openai/mujoco-py)


- Optional
  - Botorch (https://github.com/pytorch/botorch)

To run SynCMA as well as other evolutionary baselines, use the file `exp.py`. For example:

```bash
python exp.py --optimizer SynCMA --func ackley --dim 32 --eval_num 10000 --rep 100 --lam 2
```

To run TuRBO over our mentioned benchmarks, use `bo_baseline.py` inside the baseline folder:

```bash
cd baseline

python bo_baseline.py --func ackley --dim 32  --tr_num 1 --eval_num 5000 --repeat_num 30 --gpu_idx 0
```
