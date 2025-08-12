<div align="center">

# Reasoning or Memorization? Unreliable Results of Reinforcement Learning Due to Data Contamination

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)]()
[![Contribution Welcome](https://img.shields.io/badge/Contributions-welcome-blue)]()
[![Code](https://img.shields.io/badge/Code-GithubRepo-c8a4bf.svg)](https://github.com/wumingqi/LLM-Math-Evaluation) 


</div>

This repository contains reference implementation code for the experiments in our paper. As the work is still ongoing, updates may be expected in the future.

## Paper

The link to our paper is available here: https://arxiv.org/pdf/2507.10532 


The ***RandomCalculation*** dataset files are located in `random_calculation/result`. You can also manually regenerate them if needed.


## Setup

```sh
# 准备Python环境（Prepare the Python Environment）
conda create -n llm-math-evaluation python=3.10 
conda activate llm-math-evaluation

pip install -r requirements.txt
pip install flash_attn==2.7.0.post2
pip install rogue
```

## Uasge
```sh
# 评估LLM的数学能力（Evaluate the mathematical ability of LLMs）
cd math_evaluation
bash run_batch_task_math_qwen2.5.sh
# 汇总结果（Summarize Results）
python sum_metrics.py 

# 生成RandomCalculation数据集（Generate the RandomCalculation dataset）
cd random_calculation
python generate_datasets.py
```


## Acknowledgments

The code used for answer scoring is sourced from [https://github.com/ruixin31/Spurious_Rewards/](https://github.com/ruixin31/Spurious_Rewards/). We thank the authors for their valuable work.

## Citation

```bibtex
@misc{wu2025reasoningmemorizationunreliableresults,
      title={Reasoning or Memorization? Unreliable Results of Reinforcement Learning Due to Data Contamination}, 
      author={Mingqi Wu and Zhihao Zhang and Qiaole Dong and Zhiheng Xi and Jun Zhao and Senjie Jin and Xiaoran Fan and Yuhao Zhou and Yanwei Fu and Qin Liu and Songyang Zhang and Qi Zhang},
      year={2025},
      eprint={2507.10532},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2507.10532}, 
}
```