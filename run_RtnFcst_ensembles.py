import argparse
import json
import os

import numpy as np
import torch

from src.data import data_layer
from src.model.model_RtnFcst import FeedForwardModelWithNA_Return
from src.utils import deco_print


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Directory containing config_RF_<task_id>.json")
    p.add_argument("--logdir", required=True)
    p.add_argument("--task_id", required=True)
    p.add_argument("--trial_id", type=int, default=0)
    p.add_argument("--printOnConsole", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--saveLog", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--printFreq", type=int, default=128)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def validate_config_paths(config):
    required = [
        config["individual_feature_file"], config["individual_feature_file_valid"],
        config["individual_feature_file_test"], config["macro_feature_file"],
        config["macro_feature_file_valid"], config["macro_feature_file_test"],
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing dataset files:\n%s\n\nGenerate RF datasets with create_RF_data.py first."
            % "\n".join(missing)
        )


def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    config_path = os.path.join(args.config, f"config_RF_{args.task_id}.json")
    with open(config_path) as f:
        config = json.load(f)
    validate_config_paths(config)
    deco_print("Read the following in config: ")
    print(json.dumps(config, indent=4))

    deco_print("Creating data layer")
    dl = data_layer.DataInRamInputLayer(
        config["individual_feature_file"],
        pathMacroFeature=config["macro_feature_file"],
        macroIdx=config["macro_idx"],
    )
    mean_macro, std_macro = dl.getMacroFeatureMeanStd()
    dl_valid = data_layer.DataInRamInputLayer(
        config["individual_feature_file_valid"],
        pathMacroFeature=config["macro_feature_file_valid"],
        macroIdx=config["macro_idx"],
        meanMacroFeature=mean_macro, stdMacroFeature=std_macro,
    )
    dl_test = data_layer.DataInRamInputLayer(
        config["individual_feature_file_test"],
        pathMacroFeature=config["macro_feature_file_test"],
        macroIdx=config["macro_idx"],
        meanMacroFeature=mean_macro, stdMacroFeature=std_macro,
    )
    if config["weighted_loss"]:
        loss_weight = dl.getDateCountList()
        loss_weight_valid = dl_valid.getDateCountList()
        loss_weight_test = dl_test.getDateCountList()
    else:
        loss_weight = loss_weight_valid = loss_weight_test = None
    deco_print("Data layer created")

    logdir_trial = os.path.join(args.logdir, f"RF_{args.task_id}_Trial_{args.trial_id}")
    os.makedirs(logdir_trial, exist_ok=True)

    model = FeedForwardModelWithNA_Return(config, "train")
    deco_print(f"Model on device: {model._device}")

    sharpe_train, sharpe_valid, sharpe_test = model.train_model(
        dl, dl_valid, logdir_trial,
        loss_weight=loss_weight, loss_weight_valid=loss_weight_valid,
        dl_test=dl_test, loss_weight_test=loss_weight_test,
        printOnConsole=args.printOnConsole, printFreq=args.printFreq,
        saveLog=args.saveLog,
    )

    idx_best = int(np.array(sharpe_valid).argmax())
    deco_print(
        "Return-Forecast Sharpe (best valid epoch %d): "
        "Train %.3f  Valid %.3f  Test %.3f"
        % (idx_best, sharpe_train[idx_best], sharpe_valid[idx_best], sharpe_test[idx_best])
    )


if __name__ == "__main__":
    main()
