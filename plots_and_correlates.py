import numpy as np
from scipy import stats
from matplotlib import pyplot as plt


def get_correlation(lce_estimate, lce_actual):
    lce_estimate = np.array(lce_estimate)
    lce_actual = np.array(lce_actual)

    pos_mask = lce_actual > 0

    lce_spearmanr_all = stats.spearmanr(lce_estimate, lce_actual)
    lce_spearmanr_pos = stats.spearmanr(lce_estimate[pos_mask], lce_actual[pos_mask])

    return lce_spearmanr_all, lce_spearmanr_pos


def plot_lce_estimate_and_correlation(
    lce_estimate_name,
    system_name,
    control_param_name,
    lce_estimate,
    lce_actual,
    control_params,
    sequence_length,
    logy=False,
    show_plot=True,
    sharey=True,
    plot_actual=False,
    dpi=300,
):
    lce_estimate = np.array(lce_estimate)
    lce_actual = np.array(lce_actual)

    pos_mask = lce_actual > 0
    num_samples = len(lce_actual)

    count_finite = np.sum(np.isfinite(lce_estimate))
    count_all = len(lce_estimate)
    count_finite_pos = np.sum(np.isfinite(lce_estimate[pos_mask]))
    count_all_pos = np.sum(pos_mask)

    lce_spearmanr_all = stats.spearmanr(lce_estimate, lce_actual)
    lce_spearmanr_pos = stats.spearmanr(lce_estimate[pos_mask], lce_actual[pos_mask])

    if show_plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=sharey, figsize=(12, 6), dpi=dpi)

        ax1.plot(
            control_params,
            lce_estimate,
            lw=0.9,
            label=lce_estimate_name,
        )
        if plot_actual:
            ax1.plot(
                control_params,
                lce_actual,
                lw=0.9,
                label="$\lambda_{\max}$ (Benettin)",
                c="orange"
            )
            ax1.axhline(0, linestyle="--", c="red", lw=0.75)
            ax1.set_ylabel("$\lambda_{\max}$")

        ax1.set_xlabel(f"{system_name} control parameter ${control_param_name}$")
        ax1.set_ylabel(lce_estimate_name)
        ax1.title.set_text(
            f"Finite estimates (all): {count_finite}/{count_all}. Finite estimates (chaos): {count_finite_pos}/{count_all_pos}."
        )
        ax1.legend()

        ax2.scatter(
            lce_actual,
            lce_estimate,
            s=2.0,
            label=lce_estimate_name,
        )
        # ax2.scatter(lce_actual, lce_actual, s=0.8, label="Benettin (true) $\lambda_{\max}$")
        ax2.axvline(0, linestyle="--", c="orange", lw=1)
        ax2.set_xlabel("Largest Lyapunov Exponent: $\lambda_{\max}$")
        ax2.set_ylabel(lce_estimate_name)
        ax2.title.set_text(
            f"Spearman correlation: {lce_spearmanr_all[0]:.3f} (all), {lce_spearmanr_pos[0]:.3f} (chaos)"
        )

        if logy:
            ax1.set_yscale("symlog")
            ax2.set_yscale("symlog")

        plt.suptitle(
            f"{lce_estimate_name} for {system_name} map with {num_samples} trajectories of length $n={sequence_length}$."
        )
        plt.tight_layout()
        plt.show()

    return {

        "correlations": (lce_spearmanr_all, lce_spearmanr_pos)
    }
