import re
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import combinations
from functools import reduce

def plot_performance_curves(df_units, performance_metrics, sorting_cases, colors, axes, lw=2.5, ls="-", alpha=1):
    for i, metric in enumerate(performance_metrics):
        df_units_sorted = df_units.sort_values(["sorting_case", metric], ascending=False)
        unit_indices = np.zeros(len(df_units_sorted), dtype=int)
        sorters, counts = np.unique(df_units_sorted.sorting_case, return_counts=True)
        df_units_sorted.loc[:, "unit_index"] = unit_indices
        for sorting_case in sorting_cases:
            df_units_sorted_sorting_case = df_units_sorted.query(f"sorting_case == '{sorting_case}'")
            df_units_sorted.loc[df_units_sorted_sorting_case.index, "unit_index"] = np.arange(len(df_units_sorted_sorting_case), dtype=int)
        ax = axes[i]
        sns.lineplot(data=df_units_sorted, x="unit_index", y=metric, hue="sorting_case", ax=ax, palette=colors, lw=lw, ls=ls, alpha=alpha,
                     hue_order=sorting_cases)
        if i > 0:
           ax.legend().remove()
        ax.set_xlabel("")
    return df_units_sorted

def plot_aggregated_results(dataframes, colors, include_string_in_pair=None, perf_color="grey", figsize=(20, 5)):
    figs = {}
    # Aggregated results
    performance_metrics = ["accuracy", "precision", "recall"]
    # aggregated_results_folder = results_folder / "aggregated"
    df_units = pd.merge(dataframes["performances"], dataframes["metrics_gt"])
    df_counts = dataframes["unit_counts"]
    df_run_times = dataframes["run_times"]
    df_metrics_sorted = dataframes["metrics_sorted"]
    df_metrics_matched = dataframes["metrics_matched"]

    sorting_cases = list(colors.keys())
    
    fig_perf, axes = plt.subplots(ncols=len(performance_metrics), figsize=figsize, sharey=True)

    df_units_sorted = plot_performance_curves(df_units, performance_metrics, sorting_cases, colors, axes, lw=2.5, ls="-")

    sns.despine(fig_perf)
    num_hybrid_units = np.max(df_units_sorted.groupby("sorting_case")["sorting_case"].count())
    
    # pairwise metric scatter
    figs_pair = {}
    dfs_merged = {}
    if len(sorting_cases) > 1:
        pairs = combinations(sorting_cases, 2)
        on = ["stream_name", "case", "probe", "gt_unit_id"]
        if include_string_in_pair is not None:
            pairs = [sorted(p) for p in pairs if include_string_in_pair in p]
        for i, pair in enumerate(pairs):
            fig_pair, axes = plt.subplots(ncols=len(performance_metrics), figsize=figsize, sharey=True, sharex=True)
            sorting_case1, sorting_case2 = pair
            dfs_to_merge = [df_units.query(f"sorting_case == '{sorting_case}'") for sorting_case in pair]
            df_merged = reduce(lambda  left, right: pd.merge(left, right, on=on, how='outer'), dfs_to_merge)
    
            mapper = {}
            for col in df_merged:
                if "_x" in col:
                    mapper[col] = col.replace("_x", f"_{sorting_case1}")
                elif "_y" in col:
                    mapper[col] = col.replace("_y", f"_{sorting_case2}")
            df_merged = df_merged.rename(columns=mapper)
    
            for i, metric in enumerate(performance_metrics):
                ax = axes[i]
                sns.scatterplot(data=df_merged, x=f"{metric}_{sorting_case1}", y=f"{metric}_{sorting_case2}", ax=ax, color=perf_color,
                                alpha=0.2, markers=[i]*len(df_merged))
                ax.set_title(metric.capitalize())
                if i > 0:
                    ax.legend().remove()
                ax.set_xlabel("")
                ax.plot([0, 1],[0, 1], color="grey", ls="--", alpha=0.5)
            axes[0].set_ylabel(sorting_case2)
            axes[1].set_xlabel(sorting_case1)
            sns.despine(fig_pair)
            figs_pair[f"{sorting_case1}__{sorting_case2}"] = fig_pair
            dfs_merged[f"{sorting_case1}__{sorting_case2}"] = df_merged

    figs["performance"] = fig_perf
    figs.update(figs_pair)
    return figs, dfs_merged


def is_notebook() -> bool:
    """Checks if Python is running in a Jupyter notebook

    Returns
    -------
    bool
        True if notebook, False otherwise
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False


### PLOTTING UTILS ###
def prettify_axes(axs, label_fs=15):
    """Makes axes prettier by removing top and right spines and fixing label fontsizes.

    Parameters
    ----------
    axs : list
        List of matplotlib axes
    label_fs : int, optional
        Label font size, by default 15
    """
    if not isinstance(axs, (list, np.ndarray)):
        axs = [axs]

    axs = np.array(axs).flatten()

    for ax in axs:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_xlabel(ax.get_xlabel(), fontsize=label_fs)
        ax.set_ylabel(ax.get_ylabel(), fontsize=label_fs)


### STATS UTILS ###
def cohen_d(x, y):
    """Computes the Cohen's d coefficient between samples x and y

    Parameters
    ----------
    x : np.array
        Sample x
    y : np.array
        Sample y

    Returns
    -------
    float
        the Cohen's d coefficient
    """
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)


def stat_test(df, column_group_by, test_columns, sig=0.01, paired=False, verbose=False):
    """Performs statistical tests and posthoc analysis (in case of multiple groups).

    If the distributions are normal with equal variance, it performs the ANOVA test and
    posthoc T-tests (parametric).
    Otherwise, the non-parametric Kruskal-Wallis and posthoc Conover's tests are used.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe
    column_group_by : str
        The categorical column used for grouping
    test_columns : list
        The columns containing real values to test for differences.
    sig : float, optional
        Significance level, by default 0.01
    paired : bool, default: False
        Whether paired tests should be used
    verbose : bool, optional
        If True output is verbose, by default False

    Returns
    -------
    dict
        The results dictionary containing, for each metric:

        - "pvalue" :  the p-value for the multiple-sample test
        - "posthoc" : DataFrame with posthoc p-values
        - "cohens": DataFrame with Cohen's d coefficients for significant posthoc results
        - "parametric": True if parametric, False if non-parametric
    """
    from scipy.stats import kruskal, f_oneway, shapiro, levene, ttest_ind, ttest_rel, mannwhitneyu, wilcoxon
    import scikit_posthocs as sp

    df_gb = df.groupby(column_group_by)
    results = {}
    parametric = True
    for metric in test_columns:
        if verbose:
            print(f"\nTesting metric {metric}\n")
        results[metric] = {}
        samples = ()
        for i, val in enumerate(np.unique(df[column_group_by])):
            df_val = df_gb.get_group(val)
            if verbose:
                print(f"Sample {i+1}: {val} - n. {len(df_val)}")
            samples += (df_val[metric].values,)
        # shapiro test for normality
        for sample in samples:
            _, pval_n = shapiro(sample)
            if pval_n < sig:
                parametric = False
                if verbose:
                    print("Non normal samples (Shapiro test): using non parametric tests")
                break
        # levene test for equal variances
        if not parametric:
            _, pval_var = levene(*samples)
            if pval_var < sig:
                if verbose:
                    print("Non equal variances (Levene test): using non parametric tests")
                parametric = False
        if len(samples) > 2:
            print_str = "Population test: "
            if not parametric:
                test_fun = kruskal
                print_str += "Kruscal-Wallis"
                if not paired:
                    ph_test = sp.posthoc_mannwhitney
                    print_str += " - posthoc Mann Withney U"
                else:
                    ph_test = sp.posthoc_wilcoxon
                    print_str += " - posthoc Wilcoxon signed-rank"
            else:
                test_fun = f_oneway
                print_str += "One-way ANOVA - posthoc T-test"
                ph_test = sp.posthoc_ttest
            if verbose:
                print(print_str)
            # run test:
            _, pval = test_fun(*samples)
            pval_round = pval
            if pval < sig:
                # compute posthoc and cohen's d
                posthoc = ph_test(df, val_col=metric, group_col=column_group_by, p_adjust='holm',
                                  sort=False)

                # here we just consider the bottom triangular matrix and just keep significant values
                pvals = np.tril(posthoc.to_numpy(), -1)
                pvals[pvals == 0] = np.nan
                pvals[pvals >= sig] = np.nan

                # cohen's d are computed only on significantly different distributions
                ph_c = pd.DataFrame(pvals, columns=posthoc.columns, index=posthoc.index)
                pval_round = ph_c.copy()
                cols = ph_c.columns.values
                cohens = ph_c.copy()
                for index, row in ph_c.iterrows():
                    val = row.values
                    ind_non_nan, = np.nonzero(~np.isnan(val))
                    for col_ind in ind_non_nan:
                        x = df_gb.get_group(index)[metric].values
                        y = df_gb.get_group(cols[col_ind])[metric].values
                        cohen = cohen_d(x, y)
                        cohens.loc[index, cols[col_ind]] = cohen
                        pval = ph_c.loc[index, cols[col_ind]]
                        if pval < 1e-10:
                            exp = -10
                        else:
                            exp = int(np.ceil(np.log10(pval)))
                        pval_round.loc[index, cols[col_ind]] = f"<1e{exp}"
                if verbose and is_notebook():
                    print("Post-hoc:\nP-values")
                    display(posthoc)
                    print("Rounded p-values")
                    display(pval_round)
                    print("Cohen's d")
                    display(cohens)
            else:
                if verbose:
                    print(f"Non significant: p-value {pval}")
                posthoc = None
                pval_round = None
                cohens = None
        else:
            posthoc = None
            print_str = "2-sample test: "
            if parametric:
                if paired:
                    test_fun = ttest_rel
                    print_str += "Paired T-student"
                else:
                    test_fun = ttest_ind
                    print_str += "Independent T-student"
            else:
                if paired:
                    test_fun = wilcoxon
                    print_str += "Wilcoxon ranked sum"       
                else:
                    test_fun = mannwhitneyu
                    print_str += "Mann-Whitney U"        
            if verbose:
                print(print_str)
            _, pval = test_fun(*samples)
            if pval < sig:
                cohens = cohen_d(*samples)
                if verbose:
                    if pval < 1e-10:
                        pval_round = "<1e-10"
                    else:
                        exp = int(np.ceil(np.log10(pval)))
                        pval_round = f"<1e{exp}"
                    if verbose:
                        print(f"P-value {pval_round} ({pval}) - effect size: {np.round(cohens, 3)}")
            else:
                if verbose:
                    print(f"Non significant: p-value {pval}")
                pval_round = None
                cohens = None

        results[metric]["pvalue"] = pval
        results[metric]["pvalue-round"] = pval_round
        results[metric]["posthoc"] = posthoc
        results[metric]["cohens"] = cohens
        results[metric]["parametric"] = parametric
        
    return results


### Pipeline stats ###
def get_all_ecephys_derived(docdb_client):
    """Get a limited set of all records from the database.

    Returns
    -------
    list[dict]
        List of records, limited to 50 entries.
    """
    filter_query = {"data_description.modality.abbreviation": "ecephys", "data_description.data_level": "derived"}
    response = docdb_client.retrieve_docdb_records(
        filter_query=filter_query,
    )
    return response

def get_duration_from_session(session_entry):
    """"""
    if session_entry["session"] is None:
        return np.nan
    elif session_entry["session"]["session_start_time"] is None or session_entry["session"]["session_end_time"] is None:
        return np.nan
    else:
        start = datetime.fromisoformat(session_entry["session"]["session_start_time"].replace('Z', '+00:00'))
        end = datetime.fromisoformat(session_entry["session"]["session_end_time"].replace('Z', '+00:00'))
        duration = end - start
        return duration.seconds

def get_unique_sessions_with_dates(processed_data):
    """
    Extract unique sessions by raw asset name with their creation dates.
    For each raw asset name, get the earliest creation date.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: raw_asset_name, created_date
    """
    sessions_data = []
    for entry in processed_data:
        # Extract raw asset name (remove "_sorted" suffix)
        raw_name = entry["data_description"]["name"][:entry["data_description"]["name"].find("_sorted")]
        if raw_name == "ecephys_session":
            continue

        # Parse creation date (remove timezone info for consistency)
        created_date = datetime.fromisoformat(entry["created"].replace('Z', '+00:00')).replace(tzinfo=None)
        
        sessions_data.append({
            'raw_asset_name': raw_name,
            'created_date': created_date,
            'full_name': entry["data_description"]["name"],
            'duration': get_duration_from_session(entry),
            'entry': entry
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(sessions_data)
    
    # Get the earliest creation date for each raw asset name
    # (in case there are multiple sorted versions of the same raw data)
    # For each raw_asset_name, get the row with the earliest created_date (keep full_name)
    idx = df.groupby('raw_asset_name')['created_date'].idxmax()
    unique_sessions = df.loc[idx].reset_index(drop=True)
    
    return unique_sessions


def parse_curation_notes(curation_notes):
    """
    
    """
    curation_output = {}
    qc_stats = {}

    if "Passing default QC" in curation_notes or \
        ("NOISE" in curation_notes and "SUA" in curation_notes and "MUA" in curation_notes):
        # Parse lines like "Passing default QC: 90/204"
        # fix missin linebreak
        curation_notes = curation_notes.replace("Noise", "\nNoise")
        for line in curation_notes.splitlines():
            
            match = re.match(r'\s*([A-Za-z ]+):\s*(\d+)\s*/\s*(\d+)', line)
            if match:
                key = match.group(1).strip().lower().replace(" ", "_")
                value = int(match.group(2))
                total = int(match.group(3))
                qc_stats[key] = value
                # Optionally, store total as well if needed: qc_stats[key + "_total"] = total
                if key == "passing_default_qc":
                    qc_stats["total_units"] = total
                    qc_stats["failing_qc"] = total - value
        if "noise" not in qc_stats and "sua" in qc_stats and "mua" in qc_stats:
            qc_stats["noise"] = total - qc_stats.get("sua", 0) - qc_stats.get("mua", 0)
        if "mua" in qc_stats and "sua" in qc_stats:
            qc_stats["neural"] = qc_stats.get("sua", 0) + qc_stats.get("mua", 0)
        # Map keys to output names
        key_map = {
            "passing_default_qc": "passing_qc",
            "failing_qc": "failing_qc",
            "noise": "noise_units",
            "sua": "sua_units",
            "mua": "mua_units",
            "neural": "neural_units",
            "total_units": "total_units"
        }
    elif "passing default qc" in curation_notes.lower():
        # old format
        # Look for a line like "200/435 passing default QC."
        match = re.search(r'(\d+)\s*/\s*(\d+)\s*passing default QC', curation_notes, re.IGNORECASE)
        if match:
            passing = int(match.group(1))
            total_units = int(match.group(2))
            qc_stats["passing_default_qc"] = passing
            qc_stats["total_units"] = total_units
            qc_stats["failing_qc"] = total_units - passing
        # Map keys to output names
        key_map = {
            "passing_default_qc": "passing_qc",
            "failing_qc": "failing_qc",
            "total_units": "total_units"
        }

    for k, v in key_map.items():
        if k in qc_stats:
            curation_output[v] = qc_stats[k]
    return curation_output