#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np

import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", None)


def check_ranges_validity(
    dfI_discr_quantile, bins, continuous_attributes, verbose=False
):
    for cont_attr in continuous_attributes:
        if verbose:
            print(
                cont_attr,
                len(dfI_discr_quantile[cont_attr].value_counts()),
                dict(dfI_discr_quantile[cont_attr].value_counts()),
                sum(dfI_discr_quantile[cont_attr].value_counts().values),
            )
        if len(dfI_discr_quantile[cont_attr].value_counts()) != bins:
            print(dict(dfI_discr_quantile[cont_attr].value_counts()))
            raise ValueError


def run_compas_experiments_quantile(
    name_output_dir="output",
    min_sup_divergences=[0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2],
    metrics=["d_fpr", "d_fnr", "d_error"],
    verbose=False,
    ouput_folder_dir=".",
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

    outputdir = os.path.join(os.path.curdir, name_output_dir, dataset_name, "quantile")

    from pathlib import Path

    Path(outputdir).mkdir(parents=True, exist_ok=True)

    from import_datasets import discretize

    continuous_attributes = ["priors_count", "length_of_stay", "age"]

    for bins in [2, 3, 4, 5, 6]:
        dfI_discr_quantile = discretize(
            dfI, bins=bins, strategy="quantile", adaptive=True
        )

        # TODO
        dfI_discr_quantile[true_class_name] = dfI_discr_quantile[
            true_class_name
        ].astype(int)
        dfI_discr_quantile[pred_class_name] = dfI_discr_quantile[
            pred_class_name
        ].astype(int)

        check_ranges_validity(
            dfI_discr_quantile, bins, continuous_attributes, verbose=verbose
        )

        # ### Extract divergence

        from divexplorer_generalized.FP_DivergenceExplorer import FP_DivergenceExplorer

        fp_diver = FP_DivergenceExplorer(
            dfI_discr_quantile,
            true_class_name=true_class_name,
            predicted_class_name=pred_class_name,
        )

        quantile_type = f"quantile_{bins}"

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
                f"stats_sdiv{min_sup_divergence}_quantile.json",
            )

            with open(
                filename,
                "w",
            ) as fp:
                json.dump(stats, fp)
            for metric in metrics:
                if quantile_type not in out[metric][info_list[0]]:
                    for info_i in info_list:
                        out[metric][info_i][quantile_type] = {}
                out[metric]["FP"][quantile_type][float(min_sup_divergence)] = len(FP_fm)
                out[metric]["max"][quantile_type][float(min_sup_divergence)] = stats[
                    metric
                ]["max"]

    for metric in metrics:
        outputdir = os.path.join(
            ouput_folder_dir,
            name_output_dir,
            dataset_name,
            "quantile",
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
                f"info_{i}_quantile.json",
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
    args = parser.parse_args()

    run_compas_experiments_quantile(
        name_output_dir=args.name_output_dir,
        min_sup_divergences=args.min_sup_divs,
        # show_fig=args.show_fig,
    )
