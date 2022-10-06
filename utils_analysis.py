def get_filename(
    input_dir,
    dataset_name,
    target_metric_tree,
    discr_support,
    gen_type,
    redundancy=False,
    type_cr="sum_abs",
):
    redundandy_type = "no_red_" if redundancy == True else ""
    type_cr_name = "" if type_cr == "sum_abs" else f"{type_cr}_"
    return f"{input_dir}/{dataset_name}_discr_{target_metric_tree}_{discr_support}_{type_cr_name}{gen_type}_{redundandy_type}tree_experiments.json"


def reverse_dict(score_i):
    if score_i is None:
        import warnings

        warnings.warn("Dict is None")
        return
    score_out = {}
    for type_discr_all in score_i:
        type_discr = (
            float(type_discr_all.replace("discr_", ""))
            if "discr_" in type_discr_all
            else type_discr_all
        )
        if type_discr not in score_out:
            score_out[type_discr] = {}
        for type_exp in score_i[type_discr_all]:
            if type_exp not in score_out[type_discr]:
                score_out[type_discr][type_exp] = {}
            for support in score_i[type_discr_all][type_exp]:
                for target_metric in score_i[type_discr_all][type_exp][support]:
                    if target_metric not in score_out[type_discr][type_exp]:
                        score_out[type_discr][type_exp][target_metric] = {}
                    for info in score_i[type_discr_all][type_exp][support][
                        target_metric
                    ]:
                        if info not in score_out[type_discr][type_exp][target_metric]:
                            score_out[type_discr][type_exp][target_metric][info] = {}

                        score_out[type_discr][type_exp][target_metric][info][
                            float(support)
                        ] = score_i[type_discr_all][type_exp][support][target_metric][
                            info
                        ]
    return score_out


def import_json(filename, verbose=False):
    import json
    from os import path

    if path.exists(filename):

        with open(filename, "r") as fp:
            value = json.load(fp)
        if verbose:
            print(f"Ex: {filename}")
            print(value)
        return value
    else:
        import warnings

        msg = f"{filename} not available"
        warnings.warn(msg)
        return None


def get_score(
    input_dir,
    dataset_name,
    target_metric_tree,
    discr_support,
    gen_type,
    redundancy=False,
    verbose=False,
    type_cr="sum_abs",
):
    filename = get_filename(
        input_dir,
        dataset_name,
        target_metric_tree,
        discr_support,
        gen_type,
        redundancy=redundancy,
        type_cr=type_cr,
    )
    data = import_json(filename, verbose=verbose)
    if data is None:  # is None
        import warnings

        msg = f"{filename} not existing"
        warnings.warn(msg)
        return None

    return reverse_dict(data)[discr_support]


def plotDicts(
    info_dicts,
    title="",
    xlabel="",
    ylabel="",
    marker=None,
    # limit=None,
    nameFig="",
    colorMap="tab10",
    sizeFig=(4, 3),
    labelSize=8,
    markersize=4,
    outside=False,
    linewidth=1.5,
    titleLegend="",
    tickF=False,
    yscale="linear",
    legendSize=5,
    y_limits=None,
):
    import matplotlib.pyplot as plt

    m_i = 0
    markers = [
        "o",
        "v",
        "^",
        "<",
        ">",
        "*",
        "8",
        "s",
        "p",
        "h",
        "H",
        "D",
        "d",
        "P",
        "X",
    ]
    for label, info_dict in info_dicts.items():
        label_name = label  # For clarity reasons
        if marker:
            plt.plot(
                list(info_dict.keys()),
                list(info_dict.values()),
                label=label_name,
                marker=markers[m_i],
                linewidth=linewidth,
                markersize=markersize,
            )
            m_i = m_i + 1
        else:
            plt.plot(list(info_dict.keys()), list(info_dict.values()), label=label_name)
    import cycler

    if colorMap:
        plt.rcParams["axes.prop_cycle"] = cycler.cycler(
            color=plt.get_cmap(colorMap).colors
        )
    else:
        import numpy as np

        # TODO increase size --> 10 to #keys  --> ok?
        color = plt.cm.winter(np.linspace(0, 1, 20))
        plt.rcParams["axes.prop_cycle"] = cycler.cycler("color", color)

    # if limit is not None:
    #     plt.ylim(top=limit)
    if tickF:
        xt = list(info_dict.keys())
        plt.xticks([xt[i] for i in range(0, len(xt)) if i == 0 or xt[i] * 100 % 5 == 0])

    if y_limits:
        # print(y_limits)
        ymin, ymax = y_limits
        ymin = ymin - ymax * 0.05
        ymax = ymax + ymax * 0.05
        plt.ylim((ymin, ymax))

    plt.rcParams["figure.figsize"], plt.rcParams["figure.dpi"] = sizeFig, 100
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.yscale(yscale)
    if outside:
        plt.legend(
            prop={"size": labelSize},
            bbox_to_anchor=(1.04, 1),
            loc="upper left",
            title=titleLegend,
            fontsize=5,
            title_fontsize=5,
        )
    else:
        plt.legend(
            prop={"size": labelSize},
            title=titleLegend,
            fontsize=legendSize,
            title_fontsize=9,
        )
    if nameFig:
        plt.savefig(nameFig, bbox_inches="tight")
    plt.show()
    plt.close()


def plotComparisonDict(
    info_dict_1,
    info_dict_2,
    metric,
    t1="",
    t2="",
    title="",
    outDirFigs=".",
    saveFig=False,
    name_fig="",
    xlabel="min_sup",
    sizeFig=(7, 4),  # (6, 4)
    y_limits=None,
):

    labelSize = 8
    markersize = 3
    linewidth = 1

    import matplotlib.pyplot as plt

    m_i = 0
    markers = [
        "o",
        "v",
        "^",
        "<",
        ">",
        "*",
        "8",
        "s",
        "p",
        "h",
        "H",
        "D",
        "d",
        "P",
        "X",
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    for label, info_dict in info_dict_1.items():
        ax1.plot(
            list(info_dict.keys()),
            list(info_dict.values()),
            label=label,
            marker=markers[m_i],
            linewidth=linewidth,
            markersize=markersize,
        )
        if label in info_dict_2:
            ax2.plot(
                list(info_dict_2[label].keys()),
                list(info_dict_2[label].values()),
                label=label,
                marker=markers[m_i],
                linewidth=linewidth,
                markersize=markersize,
            )
        m_i = m_i + 1
    # import cycler
    # plt.rcParams['axes.prop_cycle'] =cycler.cycler(color=plt.get_cmap("tab20").colors)

    plt.rcParams["figure.figsize"], plt.rcParams["figure.dpi"] = sizeFig, 100
    ax1.set_xlabel(f"{xlabel}\n\n(a)")
    ax2.set_xlabel(f"{xlabel}\n\n(b)")
    # ax1.set_yscale("log")
    # ax2.set_yscale("log")
    ax1.set_title(t1)
    ax2.set_title(t2)
    ax1.set_ylabel(f"{metric}")
    if y_limits:
        ymin, ymax = y_limits
        ymin = ymin - ymax * 0.05
        ymax = ymax + ymax * 0.05
        ax1.set_ylim([ymin, ymax])
        ax2.set_ylim([ymin, ymax])

    fig.suptitle(title)

    fig.tight_layout()  # pad=0.5)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")  # title="ε", fontsize=10)
    # plt.legend(loc="lower center", bbox_to_anchor=(-0.7, -0.6))
    saveFig = False
    if saveFig:
        plt.savefig(f"{outDirFigs}/{name_fig}.pdf", bbox_inches="tight")
    plt.show()
    plt.close()


def plot_scores_compare_gen(
    metric_plot,
    info_name_plot,
    input_dir,
    dataset_name,
    target_metric_tree,
    discr_support,
    gen_types=["no_gen", "with_gen"],
    ret=False,
    redundancy=False,
    type_cr="sum_abs",
    y_limits=None,
):

    info_plot = {}
    for gen_type in gen_types:
        score_i = get_score(
            input_dir,
            dataset_name,
            target_metric_tree,
            discr_support,
            gen_type,
            redundancy=redundancy,
            type_cr=type_cr,
        )
        if score_i is not None:
            info_plot[gen_type] = {
                type_exp: score_i[type_exp][metric_plot][info_name_plot]
                for type_exp in score_i
            }

    title = f"{metric_plot} {info_name_plot} - {discr_support}"
    if len(info_plot) != 2:
        import warnings

        warnings.warn("Failed import")
        return
    k1, k2 = list(info_plot.keys())  # gen_types[0], gen_types[1]

    plotComparisonDict(
        info_plot[k1],
        info_plot[k2],
        metric_plot,
        title=title,
        t1=k1,
        t2=k2,
        y_limits=y_limits,
    )
    if ret:
        return info_plot


def get_score_no_div_opt(score_i):
    if score_i is None:
        import warnings

        warnings.warn("Empty dict.")
        return
    score_out = {}
    for s_str in score_i:
        for metric_info in score_i[s_str]:
            if metric_info not in score_out:
                score_out[metric_info] = {}
            for info_name in score_i[s_str][metric_info]:
                if info_name not in score_out[metric_info]:
                    score_out[metric_info][info_name] = {}
                score_out[metric_info][info_name][float(s_str)] = score_i[s_str][
                    metric_info
                ][info_name]
    return score_out


def get_greater(d, min_sup):
    return {k1: {s: v for s, v in v1.items() if s >= min_sup} for k1, v1 in d.items()}


def compare_with_no_opt_discr(
    metric_plot,
    info_name_plot,
    score_metric_nodiv,
    input_dir,
    dataset_name,
    target_metric_tree,
    discr_support,
    threshold=0,
    gen_types=["no_gen", "with_gen"],
    redundancy=False,
    ret=False,
    type_cr="sum_abs",
    y_limit=None,
    abbreviate=False,
    selection_exp_type=None,
):
    info_plot = {}
    for gen_type in gen_types:
        score_i = get_score(
            input_dir,
            dataset_name,
            target_metric_tree,
            discr_support,
            gen_type,
            redundancy=redundancy,
            type_cr=type_cr,
        )
        if score_i is not None:
            info_plot[gen_type] = {
                type_exp
                if type_cr == "sum_abs"
                else f"{type_exp}_{type_cr}": score_i[type_exp][metric_plot][
                    info_name_plot
                ]
                for type_exp in score_i
                if selection_exp_type is None or type_exp in selection_exp_type
            }
        if score_metric_nodiv is not None:
            info_dict_3 = {
                type_exp: score_metric_nodiv[type_exp][metric_plot][info_name_plot]
                for type_exp in score_metric_nodiv
            }
            if gen_type in info_plot:
                info_plot[gen_type].update(info_dict_3)
            else:
                info_plot[gen_type] = info_dict_3
    for k in info_plot:
        info_plot[k] = get_greater(info_plot[k], threshold)

    title = f"{metric_plot} {info_name_plot} - {discr_support}"

    if len(info_plot) != 2:
        import warnings

        warnings.warn("Failed import")
        return

    if abbreviate:
        for k in info_plot:
            info_plot[k] = abbreviate_dict_names(info_plot[k])

    k1, k2 = list(info_plot.keys())  # gen_types[0], gen_types[1]
    plotComparisonDict(
        info_plot[k1],
        info_plot[k2],
        metric_plot,
        title=title,
        t1=k1,
        t2=k2,
        y_limits=y_limit,
    )
    if ret:
        return info_plot


# def get_max_value(info_dict):
#     import operator
#     max_v={}
#     for s in info_dict[list(info_dict.keys())[0]]:
#         dict_i={type_exp:info_dict[type_exp][s] for type_exp in info_dict}
#         if len(set([i for i in dict_i.values()]))==1:
#             max_v[s]="all"
#         else:
#             max_v[s]=max(dict_i.items(), key=operator.itemgetter(1))[0]
#     return max_v


# def get_max_value_keys(input_dict):
#     import operator
#     max_v={}
#     for s in input_dict[list(input_dict.keys())[0]]:
#         dict_i={type_exp:input_dict[type_exp][s] for type_exp in input_dict}
#         num_max=len(set(dict_i.values()))
#         if num_max==1:
#             max_v[s]=["all"]
#         else:
#             value = max(dict_i.values())
#             max_v[s]=[k for k, v in dict_i.items() if v == value]
#     return max_v


def get_limit(dict_input):
    min_v = min([min(dict_i.values()) for t, dict_i in dict_input.items()])
    max_v = max([max(dict_i.values()) for t, dict_i in dict_input.items()])
    return min_v, max_v


def get_limits(dict_input):
    min_v, max_v = 1, 0
    for gen_type in dict_input:
        min_v_tmp, max_v_tmp = get_limit(dict_input[gen_type])
        min_v = min(min_v, min_v_tmp)
        max_v = max(max_v, max_v_tmp)
    return min_v, max_v


def abbreviate_dict_names(input_dict, rep=None):
    if rep is None:
        rep = {
            "one_at_time": "1_sep",
            "all_attributes_continuous": "all_cont",
            "all_attributes": "all",
            "weighted_sum_abs": "w",
            "_sum_abs": "",
            "uniform": "unif",
            "default": "def",
            "quantile": "quant",
        }
    upd_dict = {}
    for k in input_dict:
        k_upd = k
        for k_rep, v_rep in rep.items():
            k_upd = k_upd.replace(k_rep, v_rep)
        upd_dict[k_upd] = input_dict[k]
    return upd_dict


def update_min_max(min_v, max_v, limits):
    return min(min_v, limits[0]), max(max_v, limits[1])


def get_limits_varying_info(
    metric_plot,
    info_name_plot,
    input_dir,
    dataset_name,
    target_metric_tree,
    discr_supports,
    score_metric_nodiv=None,
    gen_types=["no_gen", "with_gen"],
    redundancy=[False, True],
    type_crs=["sum_abs", "weighted_sum_abs"],
    threshold=0,
):
    min_v, max_v = None, None

    for discr_support in discr_supports:
        for redundancy in [True, False]:
            for gen_type in gen_types:
                for type_cr in type_crs:
                    score_i = get_score(
                        input_dir,
                        dataset_name,
                        target_metric_tree,
                        discr_support,
                        gen_type,
                        redundancy=redundancy,
                        type_cr=type_cr,
                    )
                    if score_i is not None:

                        info = {
                            type_exp: score_i[type_exp][metric_plot][info_name_plot]
                            for type_exp in score_i
                        }
                        if threshold > 0:
                            info = get_greater(info, threshold)
                        if min_v is None and max_v is None:
                            min_v, max_v = get_limit(info)
                        else:
                            min_v, max_v = update_min_max(min_v, max_v, get_limit(info))
                    if score_metric_nodiv is not None:
                        info_dict_3 = {
                            type_exp: score_metric_nodiv[type_exp][metric_plot][
                                info_name_plot
                            ]
                            for type_exp in score_metric_nodiv
                        }
                        if threshold > 0:
                            info = get_greater(info_dict_3, threshold)
                        if min_v is None and max_v is None:
                            min_v, max_v = get_limit(info_dict_3)
                        else:
                            min_v, max_v = update_min_max(
                                min_v, max_v, get_limit(info_dict_3)
                            )
    ylimit = [min_v, max_v] if (min_v != 1 and max_v != 0) else None
    return ylimit


def compare_with_no_opt_discr_sups(
    metric_plot,
    info_name_plot,
    score_metric_nodiv,
    input_dir,
    dataset_name,
    target_metric_tree,
    discr_supports,
    threshold=0,
    gen_types=["no_gen", "with_gen"],
    redundancy=False,
    ret=False,
    type_cr="sum_abs",
    shared_axis=False,
    abbreviate=True,
    selection_exp_type=None,
):
    if ret:
        info_plot_for_sup = {}
    ylimit = None
    if shared_axis:
        ylimit = get_limits_varying_info(
            metric_plot,
            info_name_plot,
            input_dir,
            dataset_name,
            target_metric_tree,
            discr_supports,
            score_metric_nodiv=score_metric_nodiv,
            gen_types=gen_types,
            redundancy=[redundancy],
            type_crs=[type_cr],
            threshold=threshold,
        )
    for discr_support in discr_supports:
        ret_val = compare_with_no_opt_discr(
            metric_plot,
            info_name_plot,
            score_metric_nodiv,
            input_dir,
            dataset_name,
            target_metric_tree,
            discr_support,
            threshold=threshold,
            gen_types=gen_types,
            redundancy=redundancy,
            ret=ret,
            type_cr=type_cr,
            y_limit=ylimit,
            abbreviate=abbreviate,
            selection_exp_type=selection_exp_type,
        )
        if ret:
            info_plot_for_sup[discr_support] = ret_val
    if ret:
        return info_plot_for_sup


def plot_scores_compare_criterion(
    input_dir,
    dataset_name,
    metric_plot,
    info_name_plot,
    target_metric_tree,
    discr_support,
    gen_types=["no_gen", "with_gen"],
    type_crs=["sum_abs", "weighted_sum_abs"],
    abbreviate=True,
    y_limits=None,
):
    info_plot = {}
    for gen_type in gen_types:
        for type_cr in type_crs:
            score_i = get_score(
                input_dir,
                dataset_name,
                target_metric_tree,
                discr_support,
                gen_type,
                type_cr=type_cr,
            )
            if score_i:
                if gen_type not in info_plot:
                    info_plot[gen_type] = {}
                info_plot[gen_type].update(
                    {
                        f"{type_exp}_{type_cr}": score_i[type_exp][metric_plot][
                            info_name_plot
                        ]
                        for type_exp in score_i
                    }
                )
    if info_plot:
        if abbreviate:
            for k, v in info_plot.items():
                info_plot[k] = abbreviate_dict_names(info_plot[k])
        title = f"{metric_plot} {info_name_plot} - {discr_support}"
        k1, k2 = list(info_plot.keys())
        plotComparisonDict(
            info_plot[k1],
            info_plot[k2],
            metric_plot,
            title=title,
            t1=k1,
            t2=k2,
            y_limits=y_limits,
        )
