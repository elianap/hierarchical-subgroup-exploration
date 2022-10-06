def test_wrapper(
    dfI,
    continuous_attributes,
    metric_tree,
    min_support_tree,
    type_experiment,
    min_sup_divergences=[0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1],  # 0.001, 0.005,
    windows=[10, 50, 100],
    apply_generalization=False,
    returnFP=False,
    cols_c=["class", "predicted"],
    class_map={"N": 0, "P": 1},
    FPM_type="fpgrowth",
    redundancy=False,
    type_cr="sum_abs",
    save_in_progress=False,
    verbose=False,
):

    if type(min_sup_divergences) is not list:
        min_sup_divergences = [min_sup_divergences]

    from tree_discretization import TreeDiscretization

    attributes = list(dfI.columns.drop(cols_c))

    tree_discr = TreeDiscretization()

    # ## Discretization and generalizations

    # We get the
    #
    # - generalization/taxonomy
    #
    # - discretization ranges for the continuos attributes

    generalization_dict, discretizations = tree_discr.get_tree_discretization(
        dfI,
        type_splitting=type_experiment,
        min_support=min_support_tree,
        metric=metric_tree,
        class_map=class_map,
        continuous_attributes=continuous_attributes,
        class_and_pred_names=cols_c,
        storeTree=True,
        type_cr=type_cr
        # attributes=None,
    )
    if verbose:
        tree_discr.printDiscretizationTrees()

    considerOnlyContinuos = True
    if considerOnlyContinuos:
        for k in list(generalization_dict.keys()):
            if k not in continuous_attributes:
                generalization_dict.pop(k, None)
    # ## Discretize the dataset

    # Discretize the dataset using the obtained discretization ranges

    from utils_discretization import discretizeDataset_from_relations

    df_s_discretized, discretized_attr = discretizeDataset_from_relations(
        dfI, discretizations
    )

    if verbose:
        print(discretized_attr)
        print(df_s_discretized.head())

    prefix = "_discr"
    discrete_attributes = list(set(attributes) - set(continuous_attributes))
    df_s_discretized_tree = df_s_discretized[
        discrete_attributes + discretized_attr + ["class", "predicted"]
    ]

    # # One hot encoding

    # One hot encoding of the discretized dataset
    from copy import deepcopy

    df = deepcopy(df_s_discretized_tree)

    suffix = "_discr"
    rename_cols = {i: i.split(suffix)[0] for i in discretized_attr}
    rename_cols.update({"class": "true_class"})
    df.rename(columns=rename_cols, inplace=True)

    pred_true_class_columns_ = ["true_class", "predicted"]
    attributes = df.columns.drop(pred_true_class_columns_)

    # X_one_hot = df.copy()[attributes]
    # X_one_hot = pd.get_dummies(X_one_hot, prefix_sep="=", columns=attributes)
    # X_one_hot.reset_index(drop=True, inplace=True)

    if apply_generalization:
        from utils_discretization import one_hot_encoding_attributes

        attributes_one_hot = one_hot_encoding_attributes(df[attributes])

        # # Incompatible attribute values

        # Incompatible attribute values
        # The items obtained from the same attribute are incompatible
        from utils_hierarchy import incompatible_attribute_value

        # attributes_one_hot = list(X_one_hot.columns)

        shared_attributes_incompatible = incompatible_attribute_value(
            attributes_one_hot
        )
        if verbose:
            print(shared_attributes_incompatible)

        # # Generalizations

        # Receive a dictionary of generalizations (as the one produced by the tree) and store in an object this information

        from generalized_attributes_hierarchy import Generalizations_hierarchy

        counter_id = len(attributes_one_hot)
        generalizations_list = Generalizations_hierarchy(
            counter_id, shared_attributes_incompatible
        )

        generalizations_list.add_generalizations(
            generalization_dict, attributes_one_hot  # , level_struct=False
        )

        # ## Incompatible items

        generalization_incompatibility = (
            generalizations_list.get_attributes_incompatibility()
        )

    from divexplorer_generalized.FP_DivergenceExplorer import FP_DivergenceExplorer

    # min_sup_divergence = 0.05
    scores_s = {}
    if returnFP:
        FP_list = []
    for min_sup_divergence in min_sup_divergences:
        print(f"{min_sup_divergence} - ", end=" ")
        fp_diver = FP_DivergenceExplorer(
            df,
            "true_class",
            "predicted",
            class_map=class_map,
            generalizations_obj=generalizations_list if apply_generalization else None,
        )
        FP_fm_input = fp_diver.getFrequentPatternDivergence(
            min_support=min_sup_divergence,
            metrics=["d_fpr", "d_fnr", "d_accuracy"],
            FPM_type=FPM_type,
            save_in_progress=save_in_progress,
        )

        if returnFP:
            FP_list.append(FP_fm_input)
        # if min_sup_divergence == 0.01:
        # from copy import deepcopy

        # FP_smaller = deepcopy(FP_fm)
        K = 100

        scores_s[min_sup_divergence] = get_df_stats(
            FP_fm_input, windows=windows, redundancy=redundancy
        )
    if returnFP:
        return scores_s, FP_list
    return scores_s


# TODO: integrate redundancy, perform it after the computation
def test_wrapper_v2(
    dfI,
    continuous_attributes,
    metric_tree,
    min_support_tree,
    type_experiment,
    min_sup_divergences=[0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1],  # 0.001, 0.005,
    windows=[10, 50, 100],
    apply_generalization=False,
    returnFP=False,
    cols_c=["class", "predicted"],
    class_map={"N": 0, "P": 1},
    FPM_type="fpgrowth",
    redundancy=False,
    type_cr="sum_abs",
    save_in_progress=False,
    verbose=False,
):

    if type(min_sup_divergences) is not list:
        min_sup_divergences = [min_sup_divergences]

    from tree_discretization import TreeDiscretization

    attributes = list(dfI.columns.drop(cols_c))

    tree_discr = TreeDiscretization()

    # ## Discretization and generalizations

    # We get the
    #
    # - generalization/taxonomy
    #
    # - discretization ranges for the continuos attributes

    generalization_dict, discretizations = tree_discr.get_tree_discretization(
        dfI,
        type_splitting=type_experiment,
        min_support=min_support_tree,
        metric=metric_tree,
        class_map=class_map,
        continuous_attributes=continuous_attributes,
        class_and_pred_names=cols_c,
        storeTree=True,
        type_cr=type_cr
        # attributes=None,
    )
    if verbose:
        tree_discr.printDiscretizationTrees()

    considerOnlyContinuos = True
    if considerOnlyContinuos:
        for k in list(generalization_dict.keys()):
            if k not in continuous_attributes:
                generalization_dict.pop(k, None)
    # ## Discretize the dataset

    # Discretize the dataset using the obtained discretization ranges

    from utils_discretization import discretizeDataset_from_relations

    df_s_discretized, discretized_attr = discretizeDataset_from_relations(
        dfI, discretizations
    )

    if verbose:
        print(discretized_attr)
        print(df_s_discretized.head())

    prefix = "_discr"
    discrete_attributes = list(set(attributes) - set(continuous_attributes))
    df_s_discretized_tree = df_s_discretized[
        discrete_attributes + discretized_attr + ["class", "predicted"]
    ]

    # # One hot encoding

    # One hot encoding of the discretized dataset
    from copy import deepcopy

    df = deepcopy(df_s_discretized_tree)

    suffix = "_discr"
    rename_cols = {i: i.split(suffix)[0] for i in discretized_attr}
    rename_cols.update({"class": "true_class"})
    df.rename(columns=rename_cols, inplace=True)

    pred_true_class_columns_ = ["true_class", "predicted"]
    attributes = df.columns.drop(pred_true_class_columns_)

    # X_one_hot = df.copy()[attributes]
    # X_one_hot = pd.get_dummies(X_one_hot, prefix_sep="=", columns=attributes)
    # X_one_hot.reset_index(drop=True, inplace=True)

    if apply_generalization:
        from utils_discretization import one_hot_encoding_attributes

        attributes_one_hot = one_hot_encoding_attributes(df[attributes])

        # # Incompatible attribute values

        # Incompatible attribute values
        # The items obtained from the same attribute are incompatible
        from utils_hierarchy import incompatible_attribute_value

        # attributes_one_hot = list(X_one_hot.columns)

        shared_attributes_incompatible = incompatible_attribute_value(
            attributes_one_hot
        )
        if verbose:
            print(shared_attributes_incompatible)

        # # Generalizations

        # Receive a dictionary of generalizations (as the one produced by the tree) and store in an object this information

        from generalized_attributes_hierarchy import Generalizations_hierarchy

        counter_id = len(attributes_one_hot)
        generalizations_list = Generalizations_hierarchy(
            counter_id, shared_attributes_incompatible
        )

        generalizations_list.add_generalizations(
            generalization_dict, attributes_one_hot  # , level_struct=False
        )

        # ## Incompatible items

        generalization_incompatibility = (
            generalizations_list.get_attributes_incompatibility()
        )

    from divexplorer_generalized.FP_DivergenceExplorer import FP_DivergenceExplorer

    # min_sup_divergence = 0.05
    scores_s = {}
    if returnFP:
        FP_list = []
    min_sup_divergence = min(min_sup_divergences)
    # for min_sup_divergence in min_sup_divergences:
    print(f"{min_sup_divergence} ", end=" ")
    fp_diver = FP_DivergenceExplorer(
        df,
        "true_class",
        "predicted",
        class_map=class_map,
        generalizations_obj=generalizations_list if apply_generalization else None,
    )
    import time

    s_time = time.time()
    FP_fm_input = fp_diver.getFrequentPatternDivergence(
        min_support=min_sup_divergence,
        metrics=["d_fpr", "d_fnr", "d_accuracy"],
        FPM_type=FPM_type,
        save_in_progress=save_in_progress,
    )
    time_ex = round((time.time() - s_time), 1)
    print(f"({time_ex}) - ", end=" ")
    scores_s[min_sup_divergence] = get_df_stats(
        FP_fm_input, windows=windows, redundancy=redundancy
    )

    for min_sup_thr in min_sup_divergences:
        print(f"{min_sup_thr} ", end=" ")
        FP_fm_input_threshold = FP_fm_input.loc[FP_fm_input["support"] >= min_sup_thr]
        if returnFP:
            FP_list.append(FP_fm_input_threshold)

        scores_s[min_sup_thr] = get_df_stats(
            FP_fm_input_threshold, windows=windows, redundancy=redundancy
        )
    if returnFP:
        return scores_s, FP_list
    return scores_s


# TODO: 11 - integrate redundancy, perform it after the computation
def test_wrapper_v3(
    dfI,
    continuous_attributes,
    metric_tree,
    min_support_tree,
    type_experiment,
    min_sup_divergences=[0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1],  # 0.001, 0.005,
    windows=[10, 50, 100],
    apply_generalization=False,
    returnFP=False,
    cols_c=["class", "predicted"],
    class_map={"N": 0, "P": 1},
    FPM_type="fpgrowth",
    redundancy=False,
    type_criterion="sum_abs",
    metrics_divergence=["d_fpr", "d_fnr", "d_accuracy", "d_error"],
    save_in_progress=False,
    verbose=False,
):

    if type(min_sup_divergences) is not list:
        min_sup_divergences = [min_sup_divergences]

    from tree_discretization import TreeDiscretization

    attributes = list(dfI.columns.drop(cols_c))

    tree_discr = TreeDiscretization()

    # ## Discretization and generalizations

    # We get the
    #
    # - generalization/taxonomy
    #
    # - discretization ranges for the continuos attributes

    generalization_dict, discretizations = tree_discr.get_tree_discretization(
        dfI,
        type_splitting=type_experiment,
        min_support=min_support_tree,
        metric=metric_tree,
        class_map=class_map,
        continuous_attributes=continuous_attributes,
        class_and_pred_names=cols_c,
        storeTree=True,
        type_criterion=type_criterion
        # attributes=None,
    )
    if verbose:
        tree_discr.printDiscretizationTrees()

    considerOnlyContinuos = True
    if considerOnlyContinuos:
        for k in list(generalization_dict.keys()):
            if k not in continuous_attributes:
                generalization_dict.pop(k, None)
    # ## Discretize the dataset

    # Discretize the dataset using the obtained discretization ranges

    from utils_discretization import discretizeDataset_from_relations

    df_s_discretized, discretized_attr = discretizeDataset_from_relations(
        dfI, discretizations
    )

    if verbose:
        print(discretized_attr)
        print(df_s_discretized.head())

    prefix = "_discr"
    discrete_attributes = list(set(attributes) - set(continuous_attributes))
    df_s_discretized_tree = df_s_discretized[
        discrete_attributes + discretized_attr + ["class", "predicted"]
    ]

    # # One hot encoding

    # One hot encoding of the discretized dataset
    from copy import deepcopy

    df = deepcopy(df_s_discretized_tree)

    suffix = "_discr"
    rename_cols = {i: i.split(suffix)[0] for i in discretized_attr}
    rename_cols.update({"class": "true_class"})
    df.rename(columns=rename_cols, inplace=True)

    pred_true_class_columns_ = ["true_class", "predicted"]
    attributes = df.columns.drop(pred_true_class_columns_)

    # X_one_hot = df.copy()[attributes]
    # X_one_hot = pd.get_dummies(X_one_hot, prefix_sep="=", columns=attributes)
    # X_one_hot.reset_index(drop=True, inplace=True)

    if apply_generalization:
        from utils_discretization import one_hot_encoding_attributes

        attributes_one_hot = one_hot_encoding_attributes(df[attributes])

        # # Incompatible attribute values

        # Incompatible attribute values
        # The items obtained from the same attribute are incompatible
        from utils_hierarchy import incompatible_attribute_value

        # attributes_one_hot = list(X_one_hot.columns)

        shared_attributes_incompatible = incompatible_attribute_value(
            attributes_one_hot
        )
        if verbose:
            print(shared_attributes_incompatible)

        # # Generalizations

        # Receive a dictionary of generalizations (as the one produced by the tree) and store in an object this information

        from generalized_attributes_hierarchy import Generalizations_hierarchy

        counter_id = len(attributes_one_hot)
        generalizations_list = Generalizations_hierarchy(
            counter_id, shared_attributes_incompatible
        )

        generalizations_list.add_generalizations(
            generalization_dict, attributes_one_hot  # , level_struct=False
        )

        # ## Incompatible items

        generalization_incompatibility = (
            generalizations_list.get_attributes_incompatibility()
        )

    from divexplorer_generalized.FP_DivergenceExplorer import FP_DivergenceExplorer

    # min_sup_divergence = 0.05
    scores_s = {}
    if returnFP:
        FP_list = []
    min_sup_divergence = min(min_sup_divergences)
    # for min_sup_divergence in min_sup_divergences:
    print(f"{min_sup_divergence} ", end=" ")
    fp_diver = FP_DivergenceExplorer(
        df,
        "true_class",
        "predicted",
        class_map=class_map,
        generalizations_obj=generalizations_list if apply_generalization else None,
    )
    import time

    s_time = time.time()
    FP_fm_input = fp_diver.getFrequentPatternDivergence(
        min_support=min_sup_divergence,
        metrics=metrics_divergence,
        FPM_type=FPM_type,
        save_in_progress=save_in_progress,
    )
    time_ex = round((time.time() - s_time), 1)
    print(f"({time_ex}) - ", end=" ")
    scores_s[min_sup_divergence] = get_df_stats(
        FP_fm_input,
        windows=windows,
        redundancy=redundancy,
        metrics=metrics_divergence,
    )

    for min_sup_thr in min_sup_divergences:
        print(f"{min_sup_thr} ", end=" ")
        FP_fm_input_threshold = FP_fm_input.loc[FP_fm_input["support"] >= min_sup_thr]
        if returnFP:
            FP_list.append(FP_fm_input_threshold)

        scores_s[min_sup_thr] = get_df_stats(
            FP_fm_input_threshold,
            windows=windows,
            redundancy=redundancy,
            metrics=metrics_divergence,
        )
    if returnFP:
        return scores_s, FP_list
    return scores_s


# TODO: 11 - integrate redundancy, perform it after the computation
def test_wrapper_v4(
    dfI,
    continuous_attributes,
    metric_tree,
    min_support_tree,
    type_experiment,
    min_sup_divergences=[0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1],  # 0.001, 0.005,
    windows=[10, 50, 100],
    apply_generalization=False,
    returnFP=False,
    cols_c=["class", "predicted"],
    class_map={"N": 0, "P": 1},
    FPM_type="fpgrowth",
    redundancy=False,
    type_criterion="sum_abs",
    metrics_divergence=["d_fpr", "d_fnr", "d_accuracy", "d_error"],
    save_in_progress=False,
    injected_biases=None,
    verbose=False,
    minimal_gain=None,
):

    if type(min_sup_divergences) is not list:
        min_sup_divergences = [min_sup_divergences]

    from tree_discretization import TreeDiscretization

    attributes = list(dfI.columns.drop(cols_c))

    tree_discr = TreeDiscretization()

    # ## Discretization and generalizations

    # We get the
    #
    # - generalization/taxonomy
    #
    # - discretization ranges for the continuos attributes

    generalization_dict, discretizations = tree_discr.get_tree_discretization(
        dfI,
        type_splitting=type_experiment,
        min_support=min_support_tree,
        metric=metric_tree,
        class_map=class_map,
        continuous_attributes=continuous_attributes,
        class_and_pred_names=cols_c,
        storeTree=True,
        type_criterion=type_criterion,
        minimal_gain=minimal_gain
        # attributes=None,
    )
    if verbose:
        tree_discr.printDiscretizationTrees()

    considerOnlyContinuos = True
    if considerOnlyContinuos:
        for k in list(generalization_dict.keys()):
            if k not in continuous_attributes:
                generalization_dict.pop(k, None)
    # ## Discretize the dataset

    # Discretize the dataset using the obtained discretization ranges

    from utils_discretization import discretizeDataset_from_relations

    df_s_discretized, discretized_attr = discretizeDataset_from_relations(
        dfI, discretizations
    )

    if verbose:
        print(discretized_attr)
        print(df_s_discretized.head())

    if injected_biases is not None:
        from utils_check_interval_similarity import (
            check_tree_interval_similarity_with_bias,
        )

        interval_similarity, overlap_n = check_tree_interval_similarity_with_bias(
            tree_discr, injected_biases, apply_generalization=apply_generalization
        )
        print(f"interval_similarity: {interval_similarity}")

    prefix = "_discr"
    discrete_attributes = list(set(attributes) - set(continuous_attributes))
    df_s_discretized_tree = df_s_discretized[
        discrete_attributes + discretized_attr + ["class", "predicted"]
    ]

    # # One hot encoding

    # One hot encoding of the discretized dataset
    from copy import deepcopy

    df = deepcopy(df_s_discretized_tree)

    suffix = "_discr"
    rename_cols = {i: i.split(suffix)[0] for i in discretized_attr}
    rename_cols.update({"class": "true_class"})
    df.rename(columns=rename_cols, inplace=True)

    pred_true_class_columns_ = ["true_class", "predicted"]
    attributes = df.columns.drop(pred_true_class_columns_)

    # X_one_hot = df.copy()[attributes]
    # X_one_hot = pd.get_dummies(X_one_hot, prefix_sep="=", columns=attributes)
    # X_one_hot.reset_index(drop=True, inplace=True)

    if apply_generalization:
        from utils_discretization import one_hot_encoding_attributes

        attributes_one_hot = one_hot_encoding_attributes(df[attributes])

        # # Incompatible attribute values

        # Incompatible attribute values
        # The items obtained from the same attribute are incompatible
        from utils_hierarchy import incompatible_attribute_value

        # attributes_one_hot = list(X_one_hot.columns)

        shared_attributes_incompatible = incompatible_attribute_value(
            attributes_one_hot
        )
        if verbose:
            print(shared_attributes_incompatible)

        # # Generalizations

        # Receive a dictionary of generalizations (as the one produced by the tree) and store in an object this information

        from generalized_attributes_hierarchy import Generalizations_hierarchy

        counter_id = len(attributes_one_hot)
        generalizations_list = Generalizations_hierarchy(
            counter_id, shared_attributes_incompatible
        )

        generalizations_list.add_generalizations(
            generalization_dict, attributes_one_hot  # , level_struct=False
        )

        # ## Incompatible items

        generalization_incompatibility = (
            generalizations_list.get_attributes_incompatibility()
        )

    from divexplorer_generalized.FP_DivergenceExplorer import FP_DivergenceExplorer

    # min_sup_divergence = 0.05
    scores_s = {}
    if returnFP:
        FP_list = []
    min_sup_divergence = min(min_sup_divergences)
    # for min_sup_divergence in min_sup_divergences:
    print(f"{min_sup_divergence} ", end=" ")
    fp_diver = FP_DivergenceExplorer(
        df,
        "true_class",
        "predicted",
        class_map=class_map,
        generalizations_obj=generalizations_list if apply_generalization else None,
    )
    import time

    s_time = time.time()
    FP_fm_input = fp_diver.getFrequentPatternDivergence(
        min_support=min_sup_divergence,
        metrics=metrics_divergence,
        FPM_type=FPM_type,
        save_in_progress=save_in_progress,
    )
    time_ex = round((time.time() - s_time), 1)
    print(f"({time_ex}) - ", end=" ")
    scores_s[min_sup_divergence] = get_df_stats(
        FP_fm_input,
        windows=windows,
        redundancy=redundancy,
        metrics=metrics_divergence,
    )

    for min_sup_thr in min_sup_divergences:
        print(f"{min_sup_thr} ", end=" ")
        FP_fm_input_threshold = FP_fm_input.loc[FP_fm_input["support"] >= min_sup_thr]
        if returnFP:
            FP_list.append(FP_fm_input_threshold)

        scores_s[min_sup_thr] = get_df_stats(
            FP_fm_input_threshold,
            windows=windows,
            redundancy=redundancy,
            metrics=metrics_divergence,
        )
    if returnFP:
        return scores_s, FP_list
    if injected_biases is not None:
        return scores_s, interval_similarity, overlap_n
    return scores_s


# TODO: 11 - integrate redundancy, perform it after the computation
def test_wrapper_similarity(
    dfI,
    continuous_attributes,
    metric_tree,
    min_support_tree,
    type_experiment,
    min_sup_divergences=[0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1],  # 0.001, 0.005,
    apply_generalization=False,
    cols_c=["class", "predicted"],
    class_map={"N": 0, "P": 1},
    type_criterion="sum_abs",
    injected_biases=None,
    verbose=False,
    minimal_gain=None,
):

    print(injected_biases)

    if type(min_sup_divergences) is not list:
        min_sup_divergences = [min_sup_divergences]

    from tree_discretization import TreeDiscretization

    tree_discr = TreeDiscretization()

    # ## Discretization and generalizations

    # We get the
    #
    # - generalization/taxonomy
    #
    # - discretization ranges for the continuos attributes

    generalization_dict, discretizations = tree_discr.get_tree_discretization(
        dfI,
        type_splitting=type_experiment,
        min_support=min_support_tree,
        metric=metric_tree,
        class_map=class_map,
        continuous_attributes=continuous_attributes,
        class_and_pred_names=cols_c,
        storeTree=True,
        type_criterion=type_criterion,
        minimal_gain=minimal_gain
        # attributes=None,
    )
    if verbose:
        tree_discr.printDiscretizationTrees()

    considerOnlyContinuos = True
    if considerOnlyContinuos:
        for k in list(generalization_dict.keys()):
            if k not in continuous_attributes:
                generalization_dict.pop(k, None)

    if injected_biases is not None:
        from utils_check_interval_similarity import (
            check_tree_interval_similarity_with_bias,
        )

        interval_similarity, overlap_n = check_tree_interval_similarity_with_bias(
            tree_discr, injected_biases, apply_generalization=apply_generalization
        )
        print(f"interval_similarity: {interval_similarity}")

        return tree_discr, interval_similarity, overlap_n
    return tree_discr


def get_df_stats(
    FP_fm_input,
    windows=[10, 50, 100],
    metrics=["d_fpr", "d_fnr", "d_accuracy", "d_error"],
    prune_not_significant=True,  # NEW 11
    redundancy=False,
):

    stat_sign_dict_cols = {
        "d_fpr": "t_value_fp",
        "d_fnr": "t_value_fn",
        "d_accuracy": "t_value_tp_tn",
        "d_error": "t_value_fp_fn",
        "d_outcome": "t_value_outcome"
    }

    score_i = {}
    import pandas as pd

    for metric_i in metrics:
        if redundancy == False:
            FP_fm = FP_fm_input
        else:
            from divexplorer_generalized.FP_Divergence import FP_Divergence

            fp_i = FP_Divergence(FP_fm_input, metric_i)
            FP_fm = fp_i.getDivergence(th_redundancy=0)

        score_i.update(FP_fm[[metric_i]].describe().to_dict())
        if prune_not_significant:
            t_value_col = stat_sign_dict_cols[metric_i]

            stats_sign = (
                FP_fm.loc[FP_fm[t_value_col] >= 2][[metric_i]]
                .describe()
                .T.add_suffix("_sign")
                .T
            )
            score_i[metric_i].update(stats_sign.to_dict()[metric_i])

        score_i.update(
            {
                "len": FP_fm.sort_values(metric_i, ascending=False)["itemsets"]
                .apply(len)
                .describe()
                .to_dict()
            }
        )
        score_i.update(
            abs(FP_fm[[metric_i]])
            .sort_values(metric_i, ascending=False)
            .rename(columns={metric_i: f"{metric_i}_abs"})
            .describe()
            .to_dict()
        )

        for K in windows:
            score_i.update(
                FP_fm.sort_values(metric_i, ascending=False)[[metric_i]]
                .head(K)
                .rename(columns={metric_i: f"{metric_i}_top_{K}"})
                .describe()
                .to_dict()
            )
            score_i.update(
                FP_fm.sort_values(metric_i, ascending=False)[[metric_i]]
                .tail(K)
                .rename(columns={metric_i: f"{metric_i}_last_{K}"})
                .describe()
                .to_dict()
            )

            score_i.update(
                FP_fm.sort_values(metric_i, key=pd.Series.abs, ascending=False)[
                    [metric_i]
                ]
                .head(K)
                .rename(columns={metric_i: f"{metric_i}_sorted_abs_top_{K}"})
                .describe()
                .to_dict()
            )

            score_i.update(
                abs(FP_fm[[metric_i]])
                .sort_values(metric_i, ascending=False)
                .rename(columns={metric_i: f"{metric_i}_abs_top_{K}"})
                .head(K)
                .describe()
                .to_dict()
            )

            score_i.update(
                {
                    f"{metric_i}_top_{K}_len": FP_fm.sort_values(
                        metric_i, ascending=False
                    )["itemsets"]
                    .head(K)
                    .apply(len)
                    .describe()
                    .to_dict()
                }
            )
            score_i.update(
                {
                    f"{metric_i}_last_{K}_len": FP_fm.sort_values(
                        metric_i, ascending=False
                    )["itemsets"]
                    .tail(K)
                    .apply(len)
                    .describe()
                    .to_dict()
                }
            )

            score_i.update(
                {
                    f"{metric_i}_abs_top_{K}_len": FP_fm.sort_values(
                        metric_i, key=pd.Series.abs, ascending=False
                    )["itemsets"]
                    .head(K)
                    .apply(len)
                    .describe()
                    .to_dict()
                }
            )

    return score_i


def test_wrapper_divergence_score(
    dfI,
    min_sup_divergences=[0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1],  # 0.001, 0.005,
    windows=[50, 100],
    returnFP=False,
    cols_c=["class", "predicted"],
    class_map={"N": 0, "P": 1},
    FPM_type="fpgrowth",
):

    if type(min_sup_divergences) is not list:
        min_sup_divergences = [min_sup_divergences]

    from divexplorer_generalized.FP_DivergenceExplorer import FP_DivergenceExplorer

    scores_s = {}
    if returnFP:
        FP_list = []
    for min_sup_divergence in min_sup_divergences:
        print(f"{min_sup_divergence} - ", end=" ")
        fp_diver = FP_DivergenceExplorer(
            dfI,
            cols_c[0],
            cols_c[1],
            class_map=class_map,
        )
        FP_fm = fp_diver.getFrequentPatternDivergence(
            min_support=min_sup_divergence,
            metrics=["d_fpr", "d_fnr", "d_accuracy"],
            FPM_type=FPM_type,
        )

        if returnFP:
            FP_list.append(FP_fm)

        K = 100

        scores_s[min_sup_divergence] = get_df_stats(FP_fm, windows=windows)

    return scores_s
