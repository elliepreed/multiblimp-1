# MultiBLiMP

Catherine Arnett's modified implementation of [Jumelet et al. (2025) MultiBLiMP 1.0](https://arxiv.org/abs/2504.02768). Any bugs or errors are my own. 

The updates include:
* documentation on how to integrate the Hugging Face datasets into the pipeline
* code to automatically download datasets for each language
* example code for running multiblimp
* handling of model checkpoints

## Setup

You will need `minicons` installed. 

```
python3 -m venv venvs/demo; source venvs/demo/bin/activate
pip install minicons
python download_data.py --langs eng tur rus
python scripts/lm_eval/eval_model.py \
  --model catherinearnett/B-GPT_en_nl_simultaneous \
  --revision 10000 \
  --data_dir hf_cache/eng/ \
  --src_dir multiblimp \
  --results_dir bgpt_multiblimp_results/ \
  --cache_dir hf_cache/

python scripts/lm_eval/eval_model.py \
  --model catherinearnett/B-GPT_en_nl_simultaneous \
  --revision 10000 \
  --data_dir hf_cache/nld/ \
  --src_dir multiblimp \
  --results_dir bgpt_multiblimp_results/ \
  --cache_dir hf_cache/
```

## Using MultiBLiMP

There are a few differences between the `eval_model` function and the original `lm_eval` from the [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness). Instead of `model_args`, there is a `model` argument. This should be a model on Hugging Face¹. I added a `--revision` flag, which allows you to add model step. Make sure to note the naming convention for the revisions for your model. For example, the Pythia models have their steps labelled as `step1000`, `step10000`, etc.

I have uploaded a script, `get_multiblimp_scores.py`, which provides an example of how to loop over multiple models, checkpoints, and languages. 

### Notes

¹ In order to run local models, you will need to adapt the `load_hf_model` function (or create a new function) in `src/lm_eval/load_model.py`.

## Citation

```
@article{jumelet2025multiblimp,
  title={MultiBLiMP 1.0: A Massively Multilingual Benchmark of Linguistic Minimal Pairs},
  author={Jumelet, Jaap and Weissweiler, Leonie and Bisazza, Arianna},
  journal={arXiv preprint arXiv:2504.02768},
  year={2025}
}
```
