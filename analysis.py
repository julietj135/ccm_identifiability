"""Post-analysis and figures"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

PARAMETER_LABELS = [[r'$K_{A_{\beta}}$',r'$\lambda_{A_{\beta}}$',r'$A_0$'],[r'$K_{\tau_{\rho}}$',r'$\lambda_{\tau_{\rho}A_{\beta}}$',r'$\lambda_{\tau_{\rho}}$'],[r'$K_N$',r'$\lambda_{N\tau_{\rho}}$',r'$\lambda_N$']]


def _output_directory(path):
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def plot_logistic_regressions(data, output_dir=".", *, biomarker="abeta", metric="age", show=True):
    """Fit identifiability against six metrics and return the figure and fitted models."""
    output_dir = _output_directory(output_dir)
    df5 = data["df5"]
    models = {}
    fig, axs = plt.subplots(nrows=3, ncols=6, figsize=(15,7))
    age_indices = [6,7,8,9,10,11]
    biomarker_indices = [0,1,2,3,4,5]

    bioname = biomarker
    group = ["abeta", "tau", "neuro"].index(biomarker)
    names = [f"id_{3 * group + j}" for j in range(3)]
    param_names = PARAMETER_LABELS[group]
    type_names = ["Mean","Median","Standard Deviation","Range","Max","Min"]
    data_ = data[f"{biomarker}_data"]
    logistic_indices = {"age": age_indices, "biomarker": biomarker_indices}[metric]

    for idx, i in enumerate(logistic_indices):
        for j in range(len(param_names)):
            axs[j,idx].plot(data_[:,i],df5[names[j]],'.k')

            x = data_[:, i]
            y = df5[names[j]]
            valid = np.isfinite(x) & y.notna().to_numpy()
            x = x[valid]
            y = y[valid]
            if y.nunique() == 2:
                model = LogisticRegression(solver='liblinear', random_state=0)
                model.fit(x.reshape(-1, 1), y)
                models[(names[j], type_names[idx])] = model
                print("coefficients: ", model.coef_[0])
                # generate evaluation metrics
                print("Accuracy :", metrics.accuracy_score(y, model.predict(x.reshape(-1, 1))))
                print(model.score(x.reshape(-1, 1), y))
                df_new = pd.DataFrame(data=np.column_stack([x, y]),columns=[type_names[idx],param_names[j]])
                sns.regplot(x=type_names[idx], y=param_names[j], data=df_new, logistic=True, ax=axs[j,idx], scatter=False, color='red')
                axs[j,idx].set(xlabel=None)
                axs[j,idx].set(ylabel=None)
                axs[j,idx].tick_params(axis="x", labelsize=16)
                axs[j,idx].tick_params(axis="y", labelsize=16)

    for j in range(len(param_names)):
        axs[j,0].set_ylabel(param_names[j],fontsize="20")

    for i in range(len(type_names)):
        axs[-1,i].set_xlabel(type_names[i],fontsize="16")

    plt.tight_layout()
    plt.savefig(output_dir / f"{bioname}_logistic_{metric}.pdf")
    if show:
        plt.show()
    return fig, models


def plot_identifiability_by_visit_count(data, *, show=True):
    """Plot the mean number of identifiable parameters for each neuro visit count."""
    df5 = data["df5"]
    result = df5.groupby('num_N')['n_ID'].agg(['mean', 'std','count'])
    result['sem'] = result['std'] / np.sqrt(result['count'])
    fig = plt.figure()
    plt.bar(result.index, result['mean'], yerr=result['sem'], capsize=5, color='y',ecolor='k')
    plt.xlabel('Number of neuro data points')
    plt.ylabel('Avg number of identifiable params')

    # Show the graph
    if show:
        plt.show()
    return fig


def compare_identifiability_by_visit_count(data):
    """Return pairwise t-test p-values indexed by the two neuro visit counts."""
    df5 = data["df5"]
    counts = range(4, 9)
    p_values = pd.DataFrame(np.nan, index=counts, columns=counts)
    for i, first in enumerate(counts):
        for j, second in enumerate(counts):
            if first != second:
                a = df5.loc[df5["num_N"] == first, "n_ID"]
                b = df5.loc[df5["num_N"] == second, "n_ID"]
                p_values.loc[first, second] = stats.ttest_ind(a, b).pvalue
                if p_values.loc[first, second] < 0.05:
                    print(i, j)
    return p_values


def plot_visit_count_by_parameter(data):
    """Return nine figures comparing visit counts by parameter identifiability."""
    df5 = data["df5"]
    figures = []
    all_param_names = [["KBETA","lambdaBETA","A_initial"],["KPTAU",'lambdaPTAUBETA','lambdaPTAU'],["KN","lambdaNtp","lambdaN"]]
    for i in range(3):
        for j in range(3):
            name = all_param_names[i][j]
            id_col_name = 'id_{}'.format(3*i+j)
            data_col = 'num_{}'.format(["A","T","N"][i])

            yesid = df5.loc[df5[id_col_name] == 1, data_col]
            noid = df5.loc[df5[id_col_name] == 0, data_col]
            pval = stats.ttest_ind(yesid,noid).pvalue

            result = df5.groupby(id_col_name)[data_col].agg(['mean', 'std','count'])
            result['sem'] = result['std'] / np.sqrt(result['count'])

            figures.append(plt.figure(figsize=(5,5)))
            plt.bar(result.index, result['mean'], yerr=result['sem'], capsize=5, width=0.4, color=['red','blue','y'][i],ecolor='k')

            if pval < 0.5:  # Threshold used in the supplied analysis.
                print(pval)
                plt.text(.5, max(np.mean(yesid), np.mean(noid))+0.09, "*", ha='center', fontsize=16)
            plt.xlabel('{} identifiability'.format(name),fontsize=14)
            plt.ylabel('average number of {} points'.format(['abeta','tau','neuro'][i]),fontsize=14)
    return figures


def plot_identifiability_by_diagnosis(data, output_dir="."):
    """Compare each parameter's identifiability across CN, MCI, and AD."""
    df5 = data["df5"]
    output_dir = _output_directory(output_dir)
    all_param_names = PARAMETER_LABELS

    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(9,7))
    axs = axs.flatten()
    k = 0

    for i in range(3):
        for j in range(3):
            name = all_param_names[i][j]
            id_col_name = 'id_{}'.format(3*i+j)

            AD = df5.loc[df5['DX_bl'] == 'AD', id_col_name]
            CN = df5.loc[df5['DX_bl'] == 'CN', id_col_name]
            MCI = df5.loc[df5['DX_bl'] == 'MCI', id_col_name]

            ad_cn_pval = stats.ttest_ind(AD,CN).pvalue
            cn_mci_pval = stats.ttest_ind(CN,MCI).pvalue
            ad_mci_pval = stats.ttest_ind(AD,MCI).pvalue

            print(name, id_col_name)
            print("AD vs CN: ", ad_cn_pval)
            print("CN vs MCI: ", cn_mci_pval)
            print("AD vs MCI: ", ad_mci_pval)

            result = df5.groupby('DX_bl')[id_col_name].agg(['mean', 'std','count']).reindex(['CN', 'MCI', 'AD'])
            result['sem'] = result['std'] / np.sqrt(result['count'])

            axs[k].bar(['CN','MCI','AD'], result['mean'], yerr=result['sem'], capsize=5, color='skyblue',ecolor='k')
            axs[k].set_ylabel(name,fontsize=20)
            axs[k].set_ylim(0.0, 1.2)
            axs[k].tick_params(axis="x", labelsize=16)
            axs[k].tick_params(axis="y", labelsize=16)

            k+=1

    plt.tight_layout()
    plt.savefig(output_dir / "dx_barplots.pdf")
    return fig


def plot_ci_by_age_metrics(data, output_dir="."):
    """Plot average confidence interval against each of the six age metrics."""
    df5 = data["df5"]
    output_dir = _output_directory(output_dir)
    fig, axs = plt.subplots(nrows=3, ncols=6, figsize=(15,6))
    names = ["avg_CI_abeta","avg_CI_tau","avg_CI_neuro"]
    param_names = ["Avg Amyloid CI","Avg Tau CI","Avg Neuro CI"]
    type_names = ["Mean","Median","Standard Deviation","Range","Max","Min"]
    age_indices = [6,7,8,9,10,11]
    biomarker_indices = [0,1,2,3,4,5]

    data_ = [data["abeta_data"], data["tau_data"], data["neuro_data"]]

    # again change for age vs biomarker metrics
    metric_list_id = age_indices

    for idx, i in enumerate(metric_list_id):
        for j in range(len(names)):
            ci = df5[names[j]]
            mask = (~ci.isna()) & (ci != 0) & (~df5["RID"].isin([4376])) # [4376, 4224, 4404, 6661]
            mask &= np.isfinite(data_[j][:, i])
            ci_clean = ci[mask]
            data_clean = data_[j][:,i][mask.to_numpy()]

            # linear regression
            print("============={} {}==================".format(type_names[idx],names[j]))
            if len(np.unique(data_clean)) < 2:
                continue
            slope, intercept, r_value, p_value, std_err = stats.linregress(data_clean,ci_clean)
            print("Intercept:", intercept)
            print("Slope:", slope)
            print("Standard error:", std_err)
            print("p-value:", p_value)
            if p_value < 0.05:
                pcolor = "red"
            else:
                pcolor = "k"

            # compute standard error of predictions
            n = len(data_clean)
            data_clean_mean = np.mean(data_clean)
            SE = std_err * np.sqrt(1/n + (data_clean - data_clean_mean)**2 / np.sum((data_clean - data_clean_mean)**2))

            # 95% confidence interval
            y_pred = slope*data_clean + intercept
            t_val = stats.t.ppf(0.975, df=n - 2)  # two-tailed
            ci_upper = y_pred + t_val * SE
            ci_lower = y_pred - t_val * SE

            # sort x for proper plotting of CI band
            sort_idx = np.argsort(data_clean)
            x_sorted = data_clean[sort_idx]
            ci_lower = ci_lower[sort_idx]
            ci_upper = ci_upper[sort_idx]
            y_sorted = y_pred[sort_idx]

            axs[j,idx].scatter(data_clean,ci_clean,s=14,c='k')
            axs[j,idx].plot(x_sorted, y_sorted, color=pcolor)
            axs[j,idx].fill_between(x_sorted, ci_lower, ci_upper, color=pcolor, alpha=0.2)
            axs[j,idx].tick_params(axis='both', labelsize=14)

            axs[j,idx].text(0.95, 0.95,
             f"m = {slope:.2f}\n$p$-val = {p_value:.3g}",
             transform=axs[j, idx].transAxes,
             fontsize=10,
             verticalalignment='top',
             horizontalalignment="right",
             color=pcolor)

    for idx, i in enumerate(metric_list_id):
        axs[-1,idx].set_xlabel(type_names[idx],fontsize="14")

    axs[0,0].set_ylabel(param_names[0],fontsize="14")
    axs[1,0].set_ylabel(param_names[1],fontsize="14")
    axs[2,0].set_ylabel(param_names[2],fontsize="14")
    plt.tight_layout()
    # plt.savefig(output_dir / "confidence_scatter_age.pdf")
    plt.savefig(output_dir / "confidence_scatter_age_linear.pdf")
    return fig


def plot_ci_heatmaps(data, output_dir="."):
    """Return four figures and the biomarker and age slope/p-value matrices."""
    df5 = data["df5"]
    output_dir = _output_directory(output_dir)
    figures = []
    heatmaps = []
    names = ["avg_CI_abeta","avg_CI_tau","avg_CI_neuro"]
    param_names = ["Avg Amyloid CI","Avg Tau CI","Avg Neuro CI"]
    type_names = ["Mean","Median","Standard Deviation","Range","Max","Min"]
    metric_list_id = [0,1,2,3,4,5]

    slopes = [[0.068028,0.06239,1.49053,0.602687,0.08905,0.0463874],
              [-0.161618,-0.1616664,0.1547995,0.071170,-0.1359468,-0.190984],
              [-0.01387,-0.0223601,0.3209485,0.11945,0.004403,-0.03175]]

    pvals = [[0.37182,0.407301,0.01451,0.0175796,0.22532,0.55536],
              [0.46937,0.46975,0.932698,0.93819,0.51004,0.42774],
              [0.90155,0.84408,0.4728,0.54560,0.966116,0.77237]]

    values = [slopes, pvals]
    heatmaps.append({"slopes": np.asarray(slopes), "p_values": np.asarray(pvals)})

    for i in range(2):
        figures.append(plt.figure(figsize=(6,4)))
        plt.imshow(values[i])
        plt.colorbar(orientation='horizontal')
        plt.title(['Slope','P-Value'][i],fontsize=14)
        plt.xticks([0, 1, 2, 3, 4, 5], ["Mean","Median","Stdv","Range","Max","Min"])
        plt.yticks([0, 1, 2], ['Amyloid', 'Tau', 'Neuro'])
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / "heatmap_{}.pdf".format(['Slope','P-Value'][i]))

    metric_list_id = [6,7,8,9,10,11]
    slopes = np.full((3,6), np.nan)
    pvals = np.full((3,6), np.nan)
    data_ = [data["abeta_data"], data["tau_data"], data["neuro_data"]]

    for idx, i in enumerate(metric_list_id):
        for j in range(len(names)):
            ci = df5[names[j]]
            mask = (~ci.isna()) & (ci != 0) & (~df5["RID"].isin([4376])) # [4376, 4224, 4404, 6661]
            mask &= np.isfinite(data_[j][:, i])
            ci_clean = ci[mask]
            data_clean = data_[j][:,i][mask.to_numpy()]

            # linear regression
            print("============={} {}==================".format(type_names[idx],names[j]))
            if len(np.unique(data_clean)) < 2:
                continue
            slope, intercept, r_value, p_value, std_err = stats.linregress(data_clean,ci_clean)
            slopes[j][idx] = slope
            pvals[j][idx] = p_value
            print("Slope:", slope)
            print("p-value:", p_value)

    values = [slopes, pvals]
    heatmaps.append({"slopes": np.asarray(slopes), "p_values": np.asarray(pvals)})

    for i in range(2):
        figures.append(plt.figure(figsize=(6,4)))
        plt.imshow(values[i])
        plt.colorbar(orientation='horizontal')
        plt.title(['Slope','P-Value'][i],fontsize=14)
        plt.xticks([0, 1, 2, 3, 4, 5], ["Mean","Median","Stdv","Range","Max","Min"])
        plt.yticks([0, 1, 2], ['Amyloid', 'Tau', 'Neuro'])
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / "heatmap_{}_age.pdf".format(['Slope','P-Value'][i]))
    return figures, heatmaps[0], heatmaps[1]


def plot_ci_by_diagnosis(data, output_dir="."):
    """Plot average confidence intervals for each biomarker by diagnosis."""
    df5 = data["df5"]
    output_dir = _output_directory(output_dir)
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,5))

    names = ["avg_CI_abeta","avg_CI_tau","avg_CI_neuro"]
    dxs = ["CN","MCI","AD"]

    for i in range(len(names)):
        ci = df5[names[i]]
        mask = (~ci.isna()) & (ci != 0) & (~df5["RID"].isin([4375]))
        df5_clean = df5[mask]
        result = df5_clean.groupby('DX_bl')[names[i]].agg(['mean', 'std','count'])
        for j, dx1 in enumerate(dxs):
            for k, dx2 in enumerate(dxs):
                pval = stats.ttest_ind(df5_clean[names[i]][df5_clean.DX_bl == dx1],df5_clean[names[i]][df5_clean.DX_bl == dx2]).pvalue
                print(names[i],dx1,dx2,pval)
        means = result['mean'].reindex(dxs)
        stdvs = result['std'].reindex(dxs)
        axs[i].bar(['CN','MCI','AD'], means, yerr=stdvs, capsize=5, color=['r','b','y'][i],ecolor='k')
        axs[i].set_ylabel(["Average Amyloid beta CI","Average Phosphorylated Tau CI","Average Neurodegeneration CI"][i],fontsize=18)
        axs[i].tick_params(axis='both', labelsize=17)

    plt.tight_layout()
    plt.savefig(output_dir / "confidence_diagnostics.png")
    return fig


def plot_identifiability_colormap(data, output_dir=".", *, show=True):
    """Plot parameter identifiability for all subjects in the colormap workbook."""
    output_dir = _output_directory(output_dir)
    df_fig = data["df_fig"].to_numpy()

    fig = plt.figure(figsize=(12,2))
    plt.pcolormesh(df_fig[::-1,:], edgecolors='k', linewidth=0.5, cmap='binary')
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.xlabel("Subjects")

    num_subjects = df_fig.shape[1]
    xticks = list(range(num_subjects))
    xtick_labels = [str(i+1) for i in range(num_subjects)]
    N = 4  # change to 3, 4, etc., if needed
    plt.xticks(xticks[::N], xtick_labels[::N], rotation=45, ha='right')

    yticks = list(np.arange(9)+0.5)
    paramnames = [label for group in PARAMETER_LABELS for label in group][::-1]
    plt.yticks(yticks,paramnames)

    plt.tight_layout()
    plt.savefig(output_dir / "colormap.pdf")
    if show:
        plt.show()
    return fig


def run_post_analysis(data, output_dir=".", *, biomarker="abeta", metric="age", show=True):
    """Create the original plots and print their statistics from load_data() output.

    biomarker (abeta/tau/neuro) and metric (age/biomarker) select the logistic
    regression panels. The other panels use the original settings. Figures save
    to output_dir. The returned figures include the original display-only plots.
    Seaborn's logistic curves and bootstrap bands require statsmodels.
    """
    logistic_fig, models = plot_logistic_regressions(
        data, output_dir, biomarker=biomarker, metric=metric, show=show
    )
    figures = [logistic_fig, plot_identifiability_by_visit_count(data, show=show)]
    compare_identifiability_by_visit_count(data)
    figures.extend(plot_visit_count_by_parameter(data))
    figures.append(plot_identifiability_by_diagnosis(data, output_dir))
    figures.append(plot_ci_by_age_metrics(data, output_dir))
    heatmap_figures, biomarker_heatmaps, age_heatmaps = plot_ci_heatmaps(data, output_dir)
    figures.extend(heatmap_figures)
    figures.append(plot_ci_by_diagnosis(data, output_dir))
    figures.append(plot_identifiability_colormap(data, output_dir, show=show))
    return {
        "figures": figures,
        "logistic_models": models,
        "biomarker_heatmaps": biomarker_heatmaps,
        "age_heatmaps": age_heatmaps,
    }
