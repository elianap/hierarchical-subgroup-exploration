#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np

import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", None)


from mdlp.discretization import MDLP

def transform_intervals(x):
    if x[0] == -np.inf:
        return f"<={x[1]}"
    elif x[1] == np.inf:
        return f">{x[0]}"
    else:
        return f"({x[0]}-{x[1]}]"
    
def transform_MDLP(df_input, Y_col, continuous_attributes, random_state = 0):
    transformer = MDLP(random_state = random_state)
    X_disc = transformer.fit_transform(df_input[continuous_attributes], df_input[Y_col])
    df_discr = pd.DataFrame(X_disc, columns =  continuous_attributes)
    for e, c in enumerate(continuous_attributes):
        df_discr[c] = transformer.cat2intervals(X_disc, e)
        df_discr[c] = df_discr[c].apply(lambda x: transform_intervals(x))
    for c in df_input.columns:
        if c not in df_discr:
            df_discr[c] = df_input[c].copy()
    return df_discr[df_input.columns]

def run_compas_experiments_mdlp(
    name_output_dir="output",
    min_sup_divergences=[0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2],
    metrics=["d_fpr", "d_fnr", "d_error"],
    verbose=False,
    ouput_folder_dir=".",
    target_metric_mdlp = "true_class",
    random_state = 0
):
    out = {metric: {} for metric in metrics}
    info_list = ["FP", "max"]

    for metric in metrics:
        out[metric] = {k: {} for k in info_list}

    # # Dataset

    dataset_name = "compas"
    risk_class_type = True

    from import_datasets import import_process_compas

    dfI, class_map = import_process_compas(
        risk_class=risk_class_type, continuous_col=True
    )
    dfI.reset_index(drop=True, inplace=True)

    dfI["predicted"] = dfI["predicted"].replace({"Medium-Low": 0, "High": 1})
    true_class_name, pred_class_name = "class", "predicted"
    class_and_pred_names = [true_class_name, pred_class_name]
    attributes = list(dfI.columns.drop(class_and_pred_names))

    dfI = dfI[attributes + class_and_pred_names]

    import os

    outputdir = os.path.join(os.path.curdir, name_output_dir, dataset_name, "mdlp")

    from pathlib import Path

    Path(outputdir).mkdir(parents=True, exist_ok=True)

    from import_datasets import discretize

    continuous_attributes = ["priors_count", "length_of_stay", "age"]
    
    Y_col = "class"

    df_discr_mdlp = transform_MDLP(dfI, true_class_name, continuous_attributes)
    df_input = dfI.copy()
    
    if target_metric_mdlp == "d_error":
        target_col = "error"
        df_input[target_col] = (df_input[true_class_name]!=df_input[pred_class_name]).astype(int)
    elif target_metric_mdlp == "d_fpr":
        target_col = "fp"    
        df_input[target_col] = ((df_input[true_class_name]!=df_input[pred_class_name]) & df_input[pred_class_name]==1).astype(int)
    elif target_metric_mdlp == "d_fnr":
        target_col = "fn"    
        df_input[target_col] = ((df_input[true_class_name]!=df_input[pred_class_name]) & df_input[pred_class_name]==0).astype(int)
    elif target_metric_mdlp == "d_fnr":
        target_col = "fn"    
        df_input[target_col] = ((df_input[true_class_name]!=df_input[pred_class_name]) & df_input[pred_class_name]==0).astype(int)
    elif target_metric_mdlp == "true_class":
        target_col = true_class_name
    elif target_metric_mdlp == "predicted_class":
        target_col = pred_class_name
    else:
        raise ValueError(target_metric_mdlp)

    df_discr_mdlp = transform_MDLP(df_input, target_col, continuous_attributes)
    print(target_col)
    if target_col not in dfI.columns:
        df_input.drop(columns = [target_col], inplace=True)
        df_discr_mdlp.drop(columns = [target_col], inplace=True)
        
    # ### Extract divergence

    from divexplorer_generalized.FP_DivergenceExplorer import FP_DivergenceExplorer

    fp_diver = FP_DivergenceExplorer(
        df_discr_mdlp,
        true_class_name=true_class_name,
        predicted_class_name=pred_class_name,
    )


    for min_sup_divergence in min_sup_divergences:

        FP_fm = fp_diver.getFrequentPatternDivergence(
            min_support=min_sup_divergence,
            metrics=["d_fpr", "d_fnr", "d_accuracy", "d_error"],
        )

        from utils_experiments_runs import get_df_stats

        stats = get_df_stats(FP_fm)

        import json

        filename = os.path.join(
            outputdir,
            f"stats_sdiv{min_sup_divergence}_mdlp.json",
        )

        with open(
            filename,
            "w",
        ) as fp:
            json.dump(stats, fp)
        for metric in metrics:
            out[metric]["FP"][float(min_sup_divergence)] = len(FP_fm)
            out[metric]["max"][float(min_sup_divergence)] = stats[metric]["max"]

    for metric in metrics:
        outputdir = os.path.join(
            ouput_folder_dir,
            name_output_dir,
            dataset_name,
            "mdlp",
            metric,
        )
        from pathlib import Path

        Path(outputdir).mkdir(parents=True, exist_ok=True)
        if verbose:
            print(outputdir)

        import json

        for i in info_list:
            output = out[metric][i]

            filename = os.path.join(
                outputdir,
                f"info_{i}.json",
            )

            with open(
                filename,
                "w",
            ) as fp:
                json.dump(output, fp)


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name_output_dir",
        default="output_res",
        help="specify the name of the output folder",
    )

    parser.add_argument(
        "--min_sup_divs",
        nargs="*",
        type=float,
        default=[
            0.01,
            0.02,
            0.03,
            0.04,
            0.05,
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.35,
        ],
        help="specify a list of min supports of interest, with values from 0 to 1",
    )

    parser.add_argument(
        "--target_metric_mdlp",
        default="true_class",
        help="specify the target of the mdlp discretization",
    )
    parser.add_argument(
        "--random_state",
        default=0,
        help="specify the random state",
    )




    args = parser.parse_args()

    run_compas_experiments_mdlp(
        name_output_dir=args.name_output_dir,
        min_sup_divergences=args.min_sup_divs,
        target_metric_mdlp = arg.target_metric_mdlp,
        random_state = arg.random_state,
    )

