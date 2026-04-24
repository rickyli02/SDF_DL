# Phase 5-6 Review

This note is a review of the current PyTorch refactor before moving into Phase 5 (validation gates) and Phase 6 (cleanup).

I focused on three questions:

1. Do the rewritten core modules look internally consistent?
2. What actually passes today?
3. What is still risky enough that it should be reviewed before calling the migration done?

## What I checked

I reviewed the current implementations of:

- [src/model/model_base.py](/Users/ricky/Developer/SDF_DL/SDF_DL/src/model/model_base.py:1)
- [src/model/model_GAN.py](/Users/ricky/Developer/SDF_DL/SDF_DL/src/model/model_GAN.py:1)
- [src/model/model_RtnFcst.py](/Users/ricky/Developer/SDF_DL/SDF_DL/src/model/model_RtnFcst.py:1)
- [src/model/model_utils.py](/Users/ricky/Developer/SDF_DL/SDF_DL/src/model/model_utils.py:1)
- [src/data/data_layer.py](/Users/ricky/Developer/SDF_DL/SDF_DL/src/data/data_layer.py:1)
- [run.py](/Users/ricky/Developer/SDF_DL/SDF_DL/run.py:1)
- [run_RtnFcst_ensembles.py](/Users/ricky/Developer/SDF_DL/SDF_DL/run_RtnFcst_ensembles.py:1)
- [create_RF_data.py](/Users/ricky/Developer/SDF_DL/SDF_DL/create_RF_data.py:1)
- [capture_golden.py](/Users/ricky/Developer/SDF_DL/SDF_DL/capture_golden.py:1)
- [README.md](/Users/ricky/Developer/SDF_DL/SDF_DL/README.md:1)
- [tests/test_golden.py](/Users/ricky/Developer/SDF_DL/SDF_DL/tests/test_golden.py:1)

I also ran the current tests individually:

- `pytest tests/test_data_layer.py -q` -> `9 passed`
- `pytest tests/test_model_shapes.py -q` -> `6 passed`
- `pytest tests/test_sdf_properties.py -q` -> `5 passed`
- `pytest tests/test_training_step.py -q` -> `2 passed`
- `pytest tests/test_golden.py -q` -> `4 skipped`

The non-golden validation gates are in good shape. The golden gate is not.

## Bottom Line

The core PyTorch rewrite looks real, and the main smoke tests pass.

The remaining work is not just "final polish", though. The biggest unfinished area is the golden regression path: right now it is still wired to deleted TensorFlow APIs and cannot serve as a real Phase 5 acceptance gate.

The second important area is workflow cleanup: the README, notebook, and RF-data generation path still describe or depend on the old TensorFlow-era flow. That is likely to confuse the next person who tries to use the repository end to end.

## Findings

### 1. Golden regression is still not implemented end to end

This is the main blocker before Phase 5 can really be called complete.

Why this matters:

- `test_golden.py` is supposed to be the final numerical regression gate.
- Today it does not validate anything.

What is wrong:

- The test module skips if `.claude/golden/sdf_test.npy` is missing in [tests/test_golden.py](/Users/ricky/Developer/SDF_DL/SDF_DL/tests/test_golden.py:21).
- Even if those files existed, the `pytorch_outputs` fixture is still a stub that immediately skips in [tests/test_golden.py](/Users/ricky/Developer/SDF_DL/SDF_DL/tests/test_golden.py:52).
- The repo currently only has `.claude/golden/sdf_normalized_ensemble.npy`; the `sdf_test.npy` and `weights_test.npy` files are missing.

Why `capture_golden.py` does not solve that today:

- It still imports `src.tf_compat` in [capture_golden.py](/Users/ricky/Developer/SDF_DL/SDF_DL/capture_golden.py:26), but that module no longer exists in the migrated tree.
- I confirmed that directly: `python -c "import capture_golden"` fails with `ModuleNotFoundError: No module named 'src.tf_compat'`.
- It still uses the old TensorFlow calling style with `Session`, `sess`, and `randomInitialization(sess)` in [capture_golden.py](/Users/ricky/Developer/SDF_DL/SDF_DL/capture_golden.py:63) and [capture_golden.py](/Users/ricky/Developer/SDF_DL/SDF_DL/capture_golden.py:66).
- It still calls `model.getSDF(sess, dl_test)` and `model.getWeightWithData(sess, dl_test)` in [capture_golden.py](/Users/ricky/Developer/SDF_DL/SDF_DL/capture_golden.py:69) and [capture_golden.py](/Users/ricky/Developer/SDF_DL/SDF_DL/capture_golden.py:74), but the new API is sess-free.
- It passes `config["tSize_test"]` as the fourth argument to `FeedForwardModelWithNA_GAN_Ensembled` in [capture_golden.py](/Users/ricky/Developer/SDF_DL/SDF_DL/capture_golden.py:61), but the migrated constructor now expects `device` there, not `tSize`, in [src/model/model_GAN.py](/Users/ricky/Developer/SDF_DL/SDF_DL/src/model/model_GAN.py:22).

There is also a deeper conceptual issue:

- The migrated loader only knows how to load PyTorch `.pt` checkpoints in [src/model/model_base.py](/Users/ricky/Developer/SDF_DL/SDF_DL/src/model/model_base.py:56).
- The sample checkpoints under `sample_checkpoints/` are TensorFlow checkpoint files, not `.pt` files.

So before Phase 5 closes, the team needs to choose one of these paths:

- Use the legacy `tf_reference/` code to generate the golden `.npy` files, then keep the PyTorch side only for consuming them.
- Or implement a real conversion path from the TensorFlow checkpoints into PyTorch state dicts.

Without that decision, the golden regression gate is still only a placeholder.

### 2. The repository documentation is still describing the TensorFlow version

This is not just cosmetic. It affects whether another developer can run the repo correctly.

Examples:

- The README still says the repo has been adjusted to run in TensorFlow compatibility mode in [README.md](/Users/ricky/Developer/SDF_DL/SDF_DL/README.md:13).
- The setup instructions still describe a TensorFlow-oriented environment in [README.md](/Users/ricky/Developer/SDF_DL/SDF_DL/README.md:18).
- The end-to-end flow still assumes the old notebook-driven pipeline in [README.md](/Users/ricky/Developer/SDF_DL/SDF_DL/README.md:44).

If someone reads the README today, they will come away with the wrong mental model of the codebase.

This makes Phase 6 important, not optional.

### 3. The notebook and RF-data generation flow still appear to be in the old API world

This is the next most important workflow risk after the golden tests.

The notebook is still using TensorFlow directly:

- `model_GAN.ipynb` still contains `import tensorflow as tf`.
- It still instantiates the ensemble model with the old constructor shape and TensorFlow session flow.
- It still calls methods like `getWeightWithData(sess, ...)` and `getNormalizedSDFFactor(sess, ...)`.

I did not fully re-execute the notebook, but the API references found in the notebook are plainly pre-migration.

That matters because `create_RF_data.py` depends on artifacts from that flow:

- It expects `output/task_*` folders in [create_RF_data.py](/Users/ricky/Developer/SDF_DL/SDF_DL/create_RF_data.py:12).
- It expects `sharpe/SDF_normalized_ensemble.npy` in [create_RF_data.py](/Users/ricky/Developer/SDF_DL/SDF_DL/create_RF_data.py:18).

The current `run.py` does not generate that artifact by itself:

- It trains a single PyTorch model and saves `.pt` checkpoints via `train_model()` in [run.py](/Users/ricky/Developer/SDF_DL/SDF_DL/run.py:85).
- The checkpointing mechanism writes `model-*.pt` files in [src/model/model_base.py](/Users/ricky/Developer/SDF_DL/SDF_DL/src/model/model_base.py:50).

So the current end-to-end story is unclear:

- The training scripts are PyTorch.
- The notebook still looks TensorFlow.
- The RF generation script still expects notebook-era output files.

Before proceeding, it would be worth deciding whether the notebook/RF workflow is:

- being actively migrated,
- temporarily unsupported,
- or intentionally deferred.

Right now that is ambiguous.

### 4. Import-time dependencies are making tests and scripts slower and noisier than they need to be

This is lower severity than the items above, but still worth cleaning up.

`src/utils.py` imports seaborn, matplotlib, statsmodels, pandas, and scikit-learn at module import time in [src/utils.py](/Users/ricky/Developer/SDF_DL/SDF_DL/src/utils.py:1), even though the training paths mainly need only small helpers like `deco_print`, `sharpe`, `sort_by_task_id`, and `construct_long_short_portfolio`.

Effects I observed while testing:

- Matplotlib emitted cache-directory warnings.
- Test startup was noticeably slower than expected.
- A simple `import capture_golden` spent time building Matplotlib font caches before it even reached the actual failure.

This is not a correctness bug, but it is exactly the kind of friction that shows up during Phase 5 and makes debugging feel worse than it should.

### 5. There are a few cleanup leftovers that are harmless but worth deciding on

These are not blockers, but they are good candidates for Phase 6:

- `create_rnn_cell()` in [src/model/model_utils.py](/Users/ricky/Developer/SDF_DL/SDF_DL/src/model/model_utils.py:6) looks unused now that the code calls `build_rnn()` directly.
- The README and notebook still imply TensorFlow-specific runtime expectations even though `requirements.txt` is now PyTorch-based.
- The report and comments around "golden capture" still assume the old TF checkpoint execution model.

## What Looks Good

A few important things are already in place:

- The core `DataInRamInputLayer` path works and its tests pass.
- The main GAN inference API has been made sess-free:
  - `getSDF(dl)`
  - `getWeightWithData(dl, ...)`
  - `getNormalizedSDF(dl, ...)`
- The training smoke test passes, including checkpoint save/load on the new `.pt` format.
- The runner scripts have fail-fast dataset validation in [run.py](/Users/ricky/Developer/SDF_DL/SDF_DL/run.py:26) and [run_RtnFcst_ensembles.py](/Users/ricky/Developer/SDF_DL/SDF_DL/run_RtnFcst_ensembles.py:22), which is a real usability improvement.

That means the refactor is not "theory only". There is enough real implementation here to finish cleanly.

## Recommended Order Before Proceeding

If I were sequencing the remaining work, I would do it in this order:

1. Decide how golden files are supposed to be produced.
   Use `tf_reference/` to capture them, or implement conversion from TF checkpoints to `.pt`.

2. Finish `tests/test_golden.py`.
   Add a real `pytorch_outputs` fixture and make sure the files it loads actually exist.

3. Reconcile the notebook and RF-data generation workflow.
   Either migrate `model_GAN.ipynb` and `create_RF_data.py`, or clearly mark them as still legacy / deferred.

4. Update the README so it matches the code that exists today.
   This is important because the current README still describes the wrong stack.

5. Trim heavy import-time dependencies from `src/utils.py` where practical.
   This is mostly quality-of-life, but it will make testing and script usage cleaner.

## Practical Summary

If the question is "Can we move into Phase 5-6 now?", my answer is:

- Yes for the ordinary smoke tests and core PyTorch code.
- Not yet for the final migration acceptance story, because the golden regression path is still structurally incomplete.

The single most important review item before proceeding is the golden-test strategy. Everything else is secondary to that.
