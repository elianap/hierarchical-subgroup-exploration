#!/usr/bin/env python

import os

DATASET_DIRECTORY = os.path.join(os.path.curdir, "datasets")


def save_pickle(data, filename):
    import pickle

    with open(
        filename,
        "wb",
    ) as handle:
        pickle.dump(
            data,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def run_inject_bias_varying_div_support(
    name_output_dir="output_bias",
    type_criterion="weighted_sum_abs",
    type_experiment="one_at_time",
    min_support_tree=0.1,
    min_sup_divergences=[0.05],
    show_fig=False,
    id_bias_exp=0,
    minimal_gain=0.001,
    noise=False,
    save_fig=False,
    use_stored=True,
    dataset_dir=DATASET_DIRECTORY,
    print_info=False,
    compute_divergence_scores=False,
    print_discretization_tree=True,
    input_filename_bias="injected_bias_list",
    verbose=False,
):
    if noise == False:
        print(noise)
    import pandas as pd
    import numpy as np
    import os

    pd.set_option("display.max_colwidth", None)

    minimal_gain_str = f"gain_{minimal_gain}"

    if minimal_gain is "None":
        minimal_gain = None

    save_result = True

    # Dataset
    abbreviations = {}

    dataset_name = "artificial_10_10k"

    class_map = {"P": 1, "N": 0}

    true_class_name, pred_class_name = "class", "predicted"

    from import_datasets import is_dataset_available

    if use_stored and is_dataset_available(f"{dataset_name}.csv", inputDir=dataset_dir):
        df_artificial = pd.read_csv(os.path.join(dataset_dir, f"{dataset_name}.csv"))
    else:
        from import_datasets import create_artificial_dataset

        df_artificial = create_artificial_dataset()

    attributes = list(df_artificial.columns.drop(true_class_name))
    continuous_attributes = attributes

    ## TODO
    ## Read biases dict

    filename_biases = os.path.join(dataset_dir, f"{input_filename_bias}.pickle")

    import pickle

    with open(
        filename_biases,
        "rb",
    ) as fp:
        injected_biases_info = pickle.load(fp)

    ####  Inject bias
    if verbose:
        #     print("\n\n ###########################")
        #     print(id_bias_exp)
        # else:
        if (id_bias_exp % 5) == 0:
            print(id_bias_exp)
        else:
            print(id_bias_exp, end=" ")

    pattern_injected_bias, ratio, metric, support_count_bias = injected_biases_info[
        id_bias_exp
    ]

    df_analysis = df_artificial.copy()
    df_analysis[pred_class_name] = df_analysis[true_class_name]

    import random

    from inject_bias import Injected_bias

    injected_bias = Injected_bias(
        pattern_injected_bias,
        ratio=ratio,
        noise=noise,
        metric=metric,
        id_exp=id_bias_exp,
        dataset_size=len(df_analysis),
    )

    injected_bias.define_indexes_bias(df_analysis)

    df_analysis = injected_bias.flip_values_injected_bias(df_analysis)

    from sklearn.metrics import accuracy_score

    if verbose:
        print(
            "Accuracy: ", accuracy_score(df_analysis["class"], df_analysis["predicted"])
        )

    if noise:
        indexes_flip = list(df_analysis.index)

        random.Random(7).shuffle(indexes_flip)
        from inject_bias import flip_values

        df_analysis, indexes_flip = flip_values(
            df_analysis, indexes_flip, ratio=1 / 100
        )

        injected_bias.size_noise = len(indexes_flip)
        if verbose:
            print(
                "Accuracy: ",
                accuracy_score(df_analysis["class"], df_analysis["predicted"]),
            )

    metric = injected_bias.metric

    ###################################################

    ## Experiments

    id_exp = injected_bias.id_exp
    if verbose:
        print(id_exp, "\n\n")

    import os

    tree_outputdir_results = os.path.join(
        os.path.curdir,
        name_output_dir,
        "experiment_runs",
        dataset_name,
        f"{id_exp}",
    )

    if save_result:
        if verbose:
            print("Output dir results")
            print(tree_outputdir_results)
        from pathlib import Path

        Path(tree_outputdir_results).mkdir(parents=True, exist_ok=True)

    ### Save bias info
    filename = os.path.join(
        tree_outputdir_results, f"{dataset_name}_bias_info_{id_exp}.pickle"
    )

    save_pickle(injected_bias.get_summary_dictionary_bias(), filename)

    from utils_experiments_runs import test_wrapper_v4

    #### Print info

    if verbose:

        print(f"Injected bias: {injected_bias}")

    if print_info:

        print(f"Minimal gain: {minimal_gain}")
        print(f"Criterion type: {type_criterion}")
        print(f"Experiment type: {type_experiment}")

    # from utils_metric_tree import *
    if verbose:
        print(f"Min sup tree: {min_support_tree}")
    # if type_experiment == "one_at_time":
    #     if min_support_tree < 0.05:
    #         raise ValueError("TODO")

    tree_outputdir = os.path.join(
        tree_outputdir_results,
        type_experiment,
        type_criterion,
        minimal_gain_str,
        metric,
        f"min_sup_tree_{min_support_tree}",
    )

    tree_outputdir_stats = os.path.join(tree_outputdir, "stats_divergence")
    tree_outputdir_intervals = os.path.join(tree_outputdir, "stats_intervals")

    if save_result:
        from pathlib import Path

        for dir_name in [
            tree_outputdir,
            tree_outputdir_stats,
            tree_outputdir_intervals,
        ]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)

    #### Extract tree
    from tree_discretization import TreeDiscretization

    tree_discr = TreeDiscretization()

    # ## Extract tree
    generalization_dict, discretizations = tree_discr.get_tree_discretization(
        df_analysis,
        type_splitting=type_experiment,
        min_support=min_support_tree,
        metric=metric,
        class_map=class_map,
        continuous_attributes=list(continuous_attributes),
        class_and_pred_names=[true_class_name, pred_class_name],
        storeTree=True,
        type_criterion=type_criterion,
        minimal_gain=minimal_gain,
    )

    if print_discretization_tree:
        tree_discr.printDiscretizationTrees(round_v=3)
    if show_fig:
        from utils_print_tree import viz_tree

        viz_tree(
            tree_discr,
            tree_outputdir=tree_outputdir,
            suffix=f"inj_{id_exp}",
            saveFig=save_fig,
        )
    n_internal_nodes, n_leaf_nodes = tree_discr.get_number_nodes()

    with open(os.path.join(tree_outputdir, "childs.json"), "w") as fp:
        import json

        json.dump(
            {"n_internal_nodes": n_internal_nodes, "n_leaf_nodes": n_leaf_nodes}, fp
        )

    if tree_discr.trees is None:
        print(
            f"Could not be discretized {type_experiment} {type_criterion} {min_support_tree}"
        )

    for apply_generalization_flag in [False, True]:
        if verbose:
            print(f"Generalization: {apply_generalization_flag}")

        if apply_generalization_flag == False:
            type_gen = "without_gen"
        else:
            type_gen = "with_gen"

        from utils_check_interval_similarity import (
            check_tree_interval_similarity_with_bias,
        )

        if tree_discr.trees is None:
            if verbose:
                print(
                    f"Could not be discretized {type_experiment} {type_criterion} {min_support_tree}"
                )
            save_pickle(
                {0: 0.0},
                os.path.join(
                    tree_outputdir_intervals,
                    f"average_highest_f_measure_{type_gen}.pickle",
                ),
            )
            continue

        info_similarity = check_tree_interval_similarity_with_bias(
            tree_discr,
            injected_bias.list_injected_bias,
            apply_generalization=apply_generalization_flag,
        )
        if verbose:

            print(f"interval_similarity recall: {info_similarity['highest_f_measure']}")
            print(f"interval_similarity: {info_similarity['highest_f_measure_detail']}")

            print(" ---> ")
            print(
                f"average_highest_f_measure: {info_similarity['average_highest_f_measure']}"
            )
            print(" ---> ")
        # Save similarity info

        for similarity_type, similarity_dict in info_similarity.items():
            save_pickle(
                similarity_dict,
                os.path.join(
                    tree_outputdir_intervals,
                    f"{similarity_type}_{type_gen}.pickle",
                ),
            )

        if compute_divergence_scores:

            for min_sup_divergence in min_sup_divergences:

                # ## Extract divergence - 1 function

                from utils_extract_divergence_generalized import (
                    extract_divergence_generalized,
                )

                FP_fm = extract_divergence_generalized(
                    df_analysis,
                    discretizations,
                    generalization_dict,
                    continuous_attributes,
                    min_sup_divergence=min_sup_divergence,
                    apply_generalization=apply_generalization_flag,
                    true_class_name=true_class_name,
                    predicted_class_name=pred_class_name,
                    class_map=class_map,
                    FPM_type="fpgrowth",
                )

                from utils_experiments_runs import get_df_stats

                stats = get_df_stats(FP_fm)

                # Summary stats of divergence results

                save_pickle(
                    stats,
                    os.path.join(
                        tree_outputdir_stats,
                        f"stats_divergence_{type_gen}_min_sup_div_{min_sup_divergence}.pickle",
                    ),
                )


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name_output_dir",
        default="output_res_bias",
        help="specify the name of the output folder",
    )

    parser.add_argument(
        "--type_criterion",
        type=str,
        default="weighted_sum_abs",
        help='specify the experiment type among ["weighted_sum_abs", "weighted_sum_pow", "entropy"]',
    )

    parser.add_argument(
        "--min_sup_tree",
        type=float,
        default=0.1,
        help="specify the minimum support for the tree induction",
    )

    parser.add_argument(
        "--K",
        default=3,
        help="specify the minimum support for the divergence pattern extraction",
    )

    parser.add_argument(
        "--show_fig",
        action="store_true",
        help="specify show_fig to show the tree graph.",
    )

    parser.add_argument(
        "--noise",
        action="store_true",
        help="specify noise to add noise in the injection.",
    )
    parser.add_argument(
        "--save_fig",
        action="store_true",
        help="specify save_fig to save the figures.",
    )

    parser.add_argument(
        "--no_diverg_compute",
        action="store_false",
        help="specify no_diverg_compute to avoid computing divergence score.",
    )

    parser.add_argument(
        "--print_tree",
        action="store_true",
        help="specify print_tree to print the discretization tree",
    )
    parser.add_argument(
        "--type_experiment",
        default="one_at_time",
        help="specify the type of experiments to evaluate among ['one_at_time', 'all_attributes', 'all_attributes_continuous']",
    )

    parser.add_argument(
        "--minimal_gain",
        type=float,
        default=0.001,
        help="specify the minimal gain. Set to 'None' if no minimal gain",
    )

    parser.add_argument(
        "--min_sup_divs",
        nargs="*",
        type=float,
        default=[
            # 0.01,
            # 0.02,
            # 0.03,
            # 0.04,
            # 0.05,
            # 0.1,
            # 0.15,
            # 0.2,
            # 0.25,
            0.3,
            0.35,
        ],
        help="specify a list of min supports of interest, with values from 0 to 1",
    )

    parser.add_argument(
        "--dataset_dir",
        default=DATASET_DIRECTORY,
        help="specify the dataset directory",
    )

    parser.add_argument(
        "--id_bias_exp",
        default=0,
        help="specify the id of the bias injection",
    )

    args = parser.parse_args()

    import time

    start_time = time.time()
    for i in range(0, 40):
        run_inject_bias_varying_div_support(
            name_output_dir=args.name_output_dir,
            type_criterion=args.type_criterion,
            type_experiment=args.type_experiment,
            min_support_tree=args.min_sup_tree,
            min_sup_divergences=args.min_sup_divs,
            show_fig=args.show_fig,
            id_bias_exp=i,
            minimal_gain=args.minimal_gain,
            noise=args.noise,
            save_fig=args.save_fig,
            dataset_dir=args.dataset_dir,
            compute_divergence_scores=args.no_diverg_compute,
            print_discretization_tree=args.print_tree,
        )

    # print("--------> time:", round((time.time() - start_time), 2))
