#!/usr/bin/env python
# coding: utf-8

from copy import deepcopy
import pandas as pd
import numpy as np

pd.set_option("display.max_colwidth", None)


def run_adult_experiments_trees_taxonomies(
    name_output_dir="output",
    type_experiments=["one_at_time", "all_attributes"],
    type_criterion="weighted_sum_abs_reference_s",
    min_support_tree=0.1,
    min_sup_divergences=[0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2],
    metrics=["d_outcome"],
    show_fig=False,
    verbose=False,
    no_compute_divergence=False,
    minimal_gain=0,
    ouput_folder_dir=".",
    compute_ratios = False
):
    
    print(compute_ratios)
    out = {metric: {} for metric in metrics}
    info_list = ["FP", "max"]
    type_gens = ["with_gen", "without_gen"]
    for metric in metrics:
        out[metric] = {k: {} for k in info_list}
        for i in info_list:
            out[metric][i] = {k: {} for k in type_experiments}
            for type_experiment in type_experiments:
                out[metric][i][type_experiment] = {k: {} for k in type_gens}

    # # Dataset

    minimal_gain = None if minimal_gain == None else minimal_gain

    dataset_name = "adult_income_taxonomy"
    import os
    import pandas as pd

    filename_d = os.path.join(
        os.path.curdir, "datasets", "ACSPUMS", "adult_dataset_income_tax.csv"
    )
    dfI = pd.read_csv(filename_d)

    attributes = list(dfI.columns.drop("income"))

    continuous_attributes = ["AGEP", "WKHP"]

    metric = "d_outcome"
    target = "income"

    dfI = dfI[attributes + [target]]

    saveFig = False
    import os

    tree_outputdir = os.path.join(".", name_output_dir, "figures", "tree")

    if saveFig:
        from pathlib import Path

        Path(tree_outputdir).mkdir(parents=True, exist_ok=True)

    # # Tree divergence - income

    all_time_results = {}
    for type_experiment in type_experiments:
        all_time_results[type_experiment] = {}
        if verbose:
            print(f"Experiment type: {type_experiment}")
        for metric in metrics:
            time_results = {"with_gen": {}, "without_gen": {}}
            if verbose:
                print(f"Metric: {metric}")
                print(f"min_support_tree: {min_support_tree}")
                print(f"type_criterion: {type_criterion}")

            outputdir = os.path.join(
                ouput_folder_dir,
                name_output_dir,
                dataset_name,
                type_criterion,
                f"stree_{min_support_tree}",
                metric,
                type_experiment,
                f"gain_{minimal_gain}",
            )

            from pathlib import Path

            Path(outputdir).mkdir(parents=True, exist_ok=True)

            df_analyze = dfI.copy()

            import time

            start_time_tree = time.time()

            from tree_discretization_ranking import TreeDiscretization_ranking

            tree_discr = TreeDiscretization_ranking()

            # ## Extract tree
            generalization_dict, discretizations = tree_discr.get_tree_discretization(
                df_analyze,
                type_splitting=type_experiment,
                min_support=min_support_tree,
                metric=metric,
                continuous_attributes=list(continuous_attributes),
                storeTree=True,
                type_criterion=type_criterion,
                minimal_gain=minimal_gain,
                target_col=target
                # minimal_gain = 0.0015
            )

            time_results["tree_time"] = time.time() - start_time_tree
            n_internal_nodes, n_leaf_nodes = tree_discr.get_number_nodes()

            with open(os.path.join(outputdir, "childs.json"), "w") as fp:
                import json

                json.dump(
                    {
                        "n_internal_nodes": n_internal_nodes,
                        "n_leaf_nodes": n_leaf_nodes,
                    },
                    fp,
                )

            if show_fig:
                from utils_print_tree import viz_tree

                viz_tree(
                    tree_discr,
                    continuous_attributes=continuous_attributes,
                    tree_outputdir=tree_outputdir,
                    suffix=f"{type_experiment}_{type_criterion}_sd_{min_support_tree}_{metric}",
                    saveFig=saveFig,
                )

            """
            considerOnlyContinuos = True
            if considerOnlyContinuos:
                for k in list(generalization_dict.keys()):
                    if k not in continuous_attributes:
                        generalization_dict.pop(k, None)
            """

            import json

            with open(os.path.join(os.path.curdir, "datasets", "ACSPUMS", "adult_taxonomies.json"), "r") as fp:
                generalization_dict_tax = json.load(fp)
            

            generalization_dict_all = deepcopy(generalization_dict)
            generalization_dict_all.update(generalization_dict_tax)

            if no_compute_divergence:
                continue

            for min_sup_divergence in min_sup_divergences:
                if verbose:
                    print(min_sup_divergence, end=" ")

                # ## Extract divergence - 1 function

                for apply_generalization in [False, True]:
                    if apply_generalization == False:
                        type_gen = "without_gen"
                    else:
                        type_gen = "with_gen"
                    from utils_extract_divergence_generalized_ranking import (
                        extract_divergence_generalized,
                    )

                    allow_overalp = (
                        True if type_experiment == "all_attributes" else False
                    )
                    if (allow_overalp) and (apply_generalization is False):
                        continue

                    if verbose:
                        print(f"({type_gen})", end=" ")
                        print(f"({allow_overalp})", end=" ")
                    start_time_divergence = time.time()
                    FP_fm = extract_divergence_generalized(
                        df_analyze,
                        discretizations,
                        generalization_dict,
                        continuous_attributes,
                        min_sup_divergence=min_sup_divergence,
                        apply_generalization=apply_generalization,
                        target_name=target,
                        FPM_type="fpgrowth",
                        metrics_divergence=metrics,
                        allow_overalp=allow_overalp,
                        type_experiment=type_experiment,
                    )
                    time_results[type_gen][min_sup_divergence] = (
                        time.time() - start_time_divergence
                    )
                    

                    from utils_experiments_runs import get_df_stats

                    stats = get_df_stats(FP_fm, metrics=metrics)

                    import json

                    filename = os.path.join(
                        outputdir,
                        f"stats_sdiv{min_sup_divergence}_{type_gen}_{metric}.json",
                    )

                    with open(
                        filename,
                        "w",
                    ) as fp:
                        json.dump(stats, fp)

                    out[metric]["FP"][type_experiment][type_gen][
                        float(min_sup_divergence)
                    ] = len(FP_fm)
                    out[metric]["max"][type_experiment][type_gen][
                        float(min_sup_divergence)
                    ] = stats[metric]["max"]

                    if compute_ratios:
                        if len(metrics)>1:
                            import warnings
                            warnings.warn(f"We compute the ratios only for the first metric: {metrics[0]}")
                        from divexplorer_generalized.FP_Divergence import FP_Divergence
                        fp_i = FP_Divergence(FP_fm, metrics[0])
                        FP_fm_s = fp_i.getDivergence(th_redundancy=0)
                        if FP_fm_s.loc[0]["itemsets"]!=frozenset():
                            raise ValueError()
                        mean_outcome = FP_fm_s.loc[0]["outcome"]
                        FP_fm_s["ratio"] = FP_fm_s["outcome"] / mean_outcome
                        import math
                        FP_fm_s["wlogr"] = FP_fm_s["support"] * (FP_fm_s["ratio"]).apply(lambda x: math.log(x))
                        out.setdefault("ratio", {}).setdefault("min", {})\
                            .setdefault(type_experiment, {}).setdefault(type_gen, {})[float(min_sup_divergence)] = min(FP_fm_s["ratio"])
                        out.setdefault("ratio", {}).setdefault("max", {})\
                            .setdefault(type_experiment, {}).setdefault(type_gen, {})[float(min_sup_divergence)] = max(FP_fm_s["ratio"])
                        out.setdefault("wlogr", {}).setdefault("min", {})\
                            .setdefault(type_experiment, {}).setdefault(type_gen, {})[float(min_sup_divergence)] = min(FP_fm_s["wlogr"])
                        out.setdefault("wlogr", {}).setdefault("max", {})\
                            .setdefault(type_experiment, {}).setdefault(type_gen, {})[float(min_sup_divergence)] = max(FP_fm_s["wlogr"])





            if metric not in all_time_results:
                all_time_results[metric] = {}
            all_time_results[metric][type_experiment] = time_results
            if verbose:
                print()

    if no_compute_divergence:
        return

    for metric in metrics:
        outputdir = os.path.join(
            ouput_folder_dir,
            name_output_dir,
            dataset_name,
            type_criterion,
            f"stree_{min_support_tree}",
            metric,
            f"gain_{minimal_gain}",
        )
        from pathlib import Path

        Path(outputdir).mkdir(parents=True, exist_ok=True)

        import json

        filename = os.path.join(
            outputdir,
            f"info_time.json",
        )
        with open(
            filename,
            "w",
        ) as fp:
            json.dump(all_time_results[metric], fp)

        for i in info_list:
            output = out[metric][i]

            filename = os.path.join(
                outputdir,
                f"info_ALL_{i}.json",
            )

            with open(
                filename,
                "w",
            ) as fp:
                json.dump(output, fp)
        
    if compute_ratios:
        
        for ratio_type in ["ratio", "wlogr"]:
            for i in ["min", "max"]:
                output = out[ratio_type][i]
                
                filename = os.path.join(
                    outputdir,
                    f"info_ALL_{ratio_type}_{i}.json",
                )
                print(filename)
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
        "--type_criterion",
        type=str,
        default="weighted_sum_abs_reference_s",
        help='specify the experiment type among ["weighted_sum_abs_reference_s", "entropy"]',
    )

    parser.add_argument(
        "--min_sup_tree",
        type=float,
        default=0.1,
        help="specify the minimum support for the tree induction",
    )

    parser.add_argument(
        "--show_fig",
        action="store_true",
        help="specify show_fig to show the tree graph.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="specify verbose to print working in progress status.",
    )

    parser.add_argument(
        "--computeratios",
        action="store_true",
        help="specify computeratios to compute also the ratio and weighted log ratios.",
    )

    parser.add_argument(
        "--no_compute_divergence",
        action="store_true",
        help="specify no_compute_divergence to not compute the divergence scores.",
    )

    parser.add_argument(
        "--type_experiments",
        nargs="*",
        type=str,
        default=[
            "one_at_time",
            "all_attributes",
        ],  # , "all_attributes_continuous"], #"",
        help="specify the types of experiments to evaluate among ['one_at_time', 'all_attributes', 'all_attributes_continuous']",
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
        "--metrics",
        nargs="*",
        type=str,
        default=["d_outcome"],  # , "d_fnr", "d_error"]
        help="specify a list of metric of interest, ['d_fpr', 'd_fnr', 'd_error', 'd_accuracy', 'd_outcome']",
    )

    parser.add_argument(
        "--minimal_gain",
        type=float,
        default=0.0,
        help="specify the minimal_gain for the tree induction",
    )

    args = parser.parse_args()

    run_adult_experiments_trees_taxonomies(
        type_criterion=args.type_criterion,
        name_output_dir=args.name_output_dir,
        type_experiments=args.type_experiments,
        min_support_tree=args.min_sup_tree,
        min_sup_divergences=args.min_sup_divs,
        show_fig=args.show_fig,
        metrics=args.metrics,
        verbose=args.verbose,
        minimal_gain=args.minimal_gain,
        no_compute_divergence=args.no_compute_divergence,
        compute_ratios = args.computeratios
    )

