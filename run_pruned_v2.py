#!/usr/bin/env python
# coding: utf-8


import warnings
warnings.filterwarnings("ignore")


import numpy as np
import os
import pandas as pd
DATASET_DIRECTORY = os.path.join(os.path.curdir, "datasets")

# # Import data
from utils_plot import get_predefined_color_labels, abbreviateValue

def run_pruning_experiemnt(dataset_name = 'wine', min_support_tree = 0.1,
    min_sup_divergences = [0.1, 0.15, 0.2],
    type_criterion="divergence_criterion",
    type_experiment = "one_at_time",
    metric = "d_fpr",
    save = True,
    output_dir = 'output_results_2',
    saveFig = True,
    dataset_dir = DATASET_DIRECTORY,
    min_instances = 50):

    print(dataset_name)
    print(min_sup_divergences)
    print(output_dir)

    additional_taxonomy = None
 
    if dataset_name == 'wine':
        from import_process_dataset import import_process_wine, train_classifier_kv

        df, class_map, continuous_attributes = import_process_wine()
        # # Train and predict with RF classifier

        df = train_classifier_kv(df)
    elif dataset_name== "compas":
        from import_process_dataset import import_compas

        df, class_map, continuous_attributes = import_compas()

    elif dataset_name== "adult":
        from import_process_dataset import import_process_adult

        df, class_map, continuous_attributes = import_process_adult()

        df = train_classifier_kv(df, encoding = True)

    elif dataset_name == 'german':
        from import_process_dataset import import_process_german, train_classifier_kv

        df, class_map, continuous_attributes = import_process_german()
        # # Train and predict with RF classifier

        df = train_classifier_kv(df, encoding=True)

    elif dataset_name == 'online_shoppers_intention':
        from import_process_dataset import import_process_online_shoppers_intention, train_classifier_kv

        df, class_map, continuous_attributes = import_process_online_shoppers_intention()
        # # Train and predict with RF classifier

        df = train_classifier_kv(df, encoding=True)

    elif dataset_name == 'folkstables':
        from import_process_dataset import import_folkstables

        df, target, continuous_attributes,  = import_folkstables()

        import json
        import os
        with open(os.path.join(os.path.curdir, "datasets", "ACSPUMS", "adult_taxonomies.json"), "r") as fp:
            additional_taxonomy = json.load(fp)
    else:
        raise ValueError()

    # # Tree divergence

    true_class_name = "class"
    pred_class_name = "predicted"
    cols_c = [true_class_name, pred_class_name]





    df_analyze = df.copy()

    if metric == 'd_outcome':

        from tree_discretization_ranking import TreeDiscretization_ranking

        tree_discr = TreeDiscretization_ranking()

        # ## Extract tree
        generalization_dict, discretizations = tree_discr.get_tree_discretization(
            df_analyze,
            type_splitting=type_experiment,
            min_support=min_support_tree,
            metric=metric,
            continuous_attributes=list(continuous_attributes),
            type_criterion=type_criterion,
            storeTree=True,
            target_col=target
        )

        if additional_taxonomy is not None:
            generalization_dict.update(additional_taxonomy)

    else:
        from tree_discretization import TreeDiscretization

        tree_discr = TreeDiscretization()

        # ## Extract tree
        generalization_dict, discretizations = tree_discr.get_tree_discretization(
            df_analyze,
            type_splitting=type_experiment,
            min_support=min_support_tree,
            metric=metric,
            class_map=class_map,
            continuous_attributes=list(continuous_attributes),
            class_and_pred_names=cols_c,
            storeTree=True,
            type_criterion=type_criterion, 
            #minimal_gain = 0.0015
        )



    # # Extract patterns


    out_maxdiv = {}
    out_time = {}
    out_fp = {}



    
    import time

    for apply_generalization in [False, True]:
        type_gen = 'generalized' if apply_generalization else 'base'
        print(type_gen)
        for keep in [True, False]:
            if keep:
                keep_items = tree_discr.get_keep_items_associated_with_divergence()
                keep_str = "_pruned"
            else:
                keep_items = None
                keep_str = ""
            print(keep_str)
            for min_sup_divergence in min_sup_divergences:
                print(min_sup_divergence, end = " ")
                if df_analyze.shape[0] * min_sup_divergence < min_instances:
                    print(f'Skipped {int(df_analyze.shape[0] * min_sup_divergence)} lower than {min_instances}')
                    continue
                s_time = time.time()
                if metric == 'd_outcome':
                        from utils_extract_divergence_generalized_ranking import  extract_divergence_generalized
                                        
                        FP_fm = extract_divergence_generalized(
                        df_analyze,
                        discretizations,
                        generalization_dict,
                        continuous_attributes,
                        min_sup_divergence=min_sup_divergence,
                        apply_generalization=apply_generalization,
                        target_name=target,
                        metrics_divergence = [metric],
                        FPM_type="fpgrowth",
                        save_in_progress = False, 
                        keep_only_positive_divergent_items=keep_items
                    )
                else:
                    from utils_extract_divergence_generalized import  extract_divergence_generalized
                    FP_fm = extract_divergence_generalized(
                        df_analyze,
                        discretizations,
                        generalization_dict,
                        continuous_attributes,
                        min_sup_divergence=min_sup_divergence,
                        apply_generalization=apply_generalization,
                        true_class_name=true_class_name,
                        predicted_class_name=pred_class_name,
                        class_map=class_map,
                        metrics_divergence = [metric],
                        FPM_type="fpgrowth",
                        save_in_progress = False, 
                        keep_only_positive_divergent_items=keep_items
                    )

                key = type_gen + keep_str
                
                out_time.setdefault(min_sup_divergence, {})[key] = time.time()-s_time

                print(f"({(time.time()-s_time):.2f})")

                most_divergent = max(FP_fm[metric])
                out_maxdiv.setdefault(min_sup_divergence, {})[key] = most_divergent

                out_fp.setdefault(min_sup_divergence, {})[key] = len(FP_fm)

                del FP_fm


    import os
    output_fig_dir = os.path.join(os.path.curdir, output_dir, "figures", "output_performance")

    if saveFig:
        

        from pathlib import Path

        Path(output_fig_dir).mkdir(parents=True, exist_ok=True)


        abbreviations = {"one_at_time":"indiv t.", \
                        "divergence_criterion":"g$\\Delta$", "entropy":"entr"}



    color_labels = get_predefined_color_labels(abbreviations)
    lines_style = {k:"-" for k in color_labels}
    lines_style.update({k:"--" for k in color_labels if( "base" in k and abbreviations["entropy"] in k)})
    lines_style.update({k:"-." for k in color_labels if( 'base' in k and abbreviations["divergence_criterion"] in k)})



    from utils_plot import plotDicts

    size_fig = (3,3)


    for info_i, results in [('time', out_time), (f"max_{metric}", out_maxdiv), ('FP', out_fp)]:

        info_plot = {}
        for sup in sorted(results.keys()):
            for type_gen in results[sup]:
                type_gen_str = abbreviateValue(f"{type_criterion}_{type_gen}", abbreviations)
                if type_gen_str not in info_plot:
                    info_plot[type_gen_str] = {}
                info_plot[type_gen_str][float(sup)] = results[sup][type_gen]

        figure_name = os.path.join(output_fig_dir, f"{dataset_name}_stree_{min_support_tree}_{metric}_{info_i}.pdf")
        
        for type_gen_str in info_plot:
            info_plot[type_gen_str] = dict(sorted(info_plot[type_gen_str].items()))


        title, ylabel = '', ''
        if info_i == 'time':
            title = 'Execution time'
            ylabel="Execution time $(seconds)$"

        elif info_i == f"max_{metric}":
            m = metric[2:].upper()
            
            ylabel=f"Max $\\Delta_{{{m}}}$"
            title=f"Highest $\\Delta_{{{m}}}$"
           

        elif info_i == 'FP':
            ylabel="#FP"
            title="#FP" 
            
        

        plotDicts(info_plot, marker=True, \
            title=title, sizeFig=size_fig,\
                    linestyle=lines_style, color_labels=color_labels, \
            xlabel="Minimum support s",  ylabel=ylabel, labelSize=10.2,\
            outside=True,  saveFig=saveFig, nameFig = figure_name, legendSize=10.2)
            


    # # Store performance results

    if save:
        import os

        output_results = os.path.join(os.path.curdir, output_dir, 'performance')
        from pathlib import Path

        Path(output_results).mkdir(parents=True, exist_ok=True)

        conf_name = f"{dataset_name}_{metric}_{type_criterion}_{min_support_tree}"

        import json
        with open(os.path.join(output_results, f'{conf_name}_time.json'), 'w') as output_file:
            output_file.write(json.dumps(out_time))


        import json
        with open(os.path.join(output_results, f'{conf_name}_fp.json'), 'w') as output_file:
            output_file.write(json.dumps(out_fp))


        with open(os.path.join(output_results, f'{conf_name}_div.json'), 'w') as output_file:
            output_file.write(json.dumps(out_maxdiv))



import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name_output_dir",
        default="output_red",
        help="specify the name of the output folder",
    )

    parser.add_argument(
        "--dataset_dir",
        default=DATASET_DIRECTORY,
        help="specify the dataset directory",
    )

    parser.add_argument(
        "--no_show_figs",
        action="store_true",
        help="specify not_show_figures to vizualize the plots. The results are stored into the specified outpur dir.",
    )


    parser.add_argument(
        "--dataset_name",
        default="wine",
        help="specify the name of the dataset",
    )

    parser.add_argument(
        "--min_support_tree",
        type=float,
        default=0.1,
        help="specify the name of the dataset",
    )
    parser.add_argument(
        "--min_sup_divergences",
        type=float,
        nargs="*",
        default=[0.15, 0.2],
        help="specify the minimum support scores",
    )
    parser.add_argument(
        "--type_criterion",
        type=str,
        default="divergence_criterion",
        help="specify the split criterion",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default='d_fpr',
        help="specify the metric",
    )
    parser.add_argument(
        "--min_instances",
        type=int,
        default=50,
        help="specify the number of minimum instances for statistical significance",
    )

    args = parser.parse_args()

    run_pruning_experiemnt(min_support_tree = args.min_support_tree,
    min_sup_divergences = args.min_sup_divergences,
    type_criterion=args.type_criterion,
    metric = args.metric,
    #save = True,
    #saveFig = True,
        dataset_name = args.dataset_name,
        output_dir=args.name_output_dir,
        dataset_dir=args.dataset_dir,
        min_instances=args.min_instances
    )