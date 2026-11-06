import seaborn as sns
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

#%% Plotting parameters
plt.rcParams["errorbar.capsize"] = 1.0
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['svg.fonttype'] = 'none'
sns.set(font_scale=1)
sns.set_style("ticks")
plt.rcParams['axes.xmargin'] = 0.05
plt.rcParams['axes.ymargin'] = 1
plt.rcParams["mathtext.default"] = 'regular'
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['text.usetex'] = False  # Disable LaTeX rendering if not needed

plt.tick_params(axis='both', left=True, right=True, labelright=False,
                direction='out', length=2, width=2, colors='k', labelcolor='k')

params = {'axes.titlesize': 12, 'axes.labelsize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10}
plt.rcParams.update(params)

#%% Check command-line arguments
if len(sys.argv) < 7:
    print("Usage: script.py <data_file> <Kd_ubound> <n_ubound> <ymin_guess> <ymax_guess> <output_prefix>")
    sys.exit(1)

data_file = sys.argv[1]
Kd_ubound = float(sys.argv[2]) if sys.argv[2].lower() != 'inf' else np.inf
n_ubound = float(sys.argv[3]) if sys.argv[3].lower() != 'inf' else 10
ymin_guess = float(sys.argv[4]) if sys.argv[4].lower() != 'none' else 1
ymax_guess = float(sys.argv[5]) if sys.argv[5].lower() != 'none' else 1
output_prefix = sys.argv[6]

#%% Define the Hill equation and residuals function
def hill(x, Kd, n, ymin, ymax):
    return ymin + (ymax - ymin) * (x**n) / (Kd**n + x**n)

def hill_res(params, ind_conc, exp_val):
    Kd, n, ymin, ymax = params
    rpu = hill(ind_conc, Kd, n, ymin, ymax)
    return rpu - exp_val

#%% Read the data file
data = pd.read_csv(data_file)

# Initialize a single plot
plt.figure(figsize=(3, 3))

# Prepare to collect fitted results
fitted_results = []

for regulator in data['Regulator expression'].unique():
    subset = data[data['Regulator expression'] == regulator]
    subset = subset.dropna(subset=['RPU_day1', 'RPU_day2', 'RPU_day3'])

    if subset.empty:
        print(f"No data available for regulator: {regulator}. Skipping.")
        continue

    inducer_conc = subset['HSL_concentraion'].values
    RPU_mean = subset[['RPU_day1', 'RPU_day2', 'RPU_day3']].mean(axis=1)
    sd_data = subset[['RPU_day1', 'RPU_day2', 'RPU_day3']].std(axis=1)

    regulator_results = {'Regulator': regulator}

    # Fit per-day Hill curves and calculate sensitivities
    for day_col in ['RPU_day1', 'RPU_day2', 'RPU_day3']:
        RPU = subset[day_col].values
        ymin_bound = RPU.min()
        ymax_bound = RPU.max()

        params0 = [1, 2, ymin_bound, ymax_bound]
        res_lsq = least_squares(
            hill_res, params0, args=(inducer_conc, RPU),
            bounds=([0, 0, 0.9 * ymin_bound, 0.9 * ymax_bound],
                    [Kd_ubound, n_ubound, 1.2 * ymin_bound, 1.2 * ymax_bound])
        )

        Kd_day, n_day, ymin_day, ymax_day = res_lsq.x

        # Sensitivity at midpoint
        x_mid = Kd_day
        numerator = (ymax_day - ymin_day) * n_day * (Kd_day ** n_day) * (x_mid ** (n_day - 1))
        denominator = (Kd_day ** n_day + x_mid ** n_day) ** 2
        sensitivity = numerator / denominator

        day_idx = day_col.split("_")[1]
        regulator_results.update({
            f'Kd_day{day_idx}': Kd_day,
            f'n_day{day_idx}': n_day,
            f'ymin_day{day_idx}': ymin_day,
            f'ymax_day{day_idx}': ymax_day,
            f'Sensitivity_day{day_idx}': sensitivity
        })

    # Fit mean Hill curve
    params0 = [1, 2, RPU_mean.min(), RPU_mean.max()]
    res_lsq_mean = least_squares(
        hill_res, params0, args=(inducer_conc, RPU_mean),
        bounds=([0, 0, 0.9 * RPU_mean.min(), 0.9 * RPU_mean.max()],
                [Kd_ubound, n_ubound, 1.2 * RPU_mean.min(), 1.2 * RPU_mean.max()])
    )
    Kd_mean, n_mean, ymin_mean, ymax_mean = res_lsq_mean.x
    regulator_results.update({
        'Kd_mean': Kd_mean,
        'n_mean': n_mean,
        'ymin_mean': ymin_mean,
        'ymax_mean': ymax_mean
    })

    # Plot data and mean Hill curve
    color = subset['c_map'].iloc[0]
    plt.errorbar(inducer_conc, RPU_mean, yerr=sd_data, fmt='o', mfc='None', mec=color, ecolor=color,
                 alpha=0.8, markersize=4, capsize=2, elinewidth=0.75)
    x_vals = np.logspace(np.log10(subset['HSL_concentraion'].min() * 0.9),
                         np.log10(subset['HSL_concentraion'].max() * 4), num=1000)
    plt.plot(x_vals, hill(x_vals, *res_lsq_mean.x), lw=1, color=color, label=f"{regulator}")

    fitted_results.append(regulator_results)

#%% Set axis scales and labels
plt.xscale('log')
plt.yscale('log')
plt.xlim(left=data['HSL_concentraion'].min() * 0.8, right=data['HSL_concentraion'].max() * 4)
plt.ylim([10**-3, 2 * 10**1])
plt.xlabel('3OC6-HSL (µM)')
plt.ylabel(r'$P_{Lux}$ output (RPU)')

plt.gca().tick_params(which='major', direction='out', length=6, width=2)
plt.gca().tick_params(which='minor', direction='out', length=4, width=1)
plt.gca().xaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=200))
plt.gca().xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs='auto', numticks=200))
plt.gca().yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=10))
plt.gca().yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs='auto', numticks=10))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{output_prefix}_combined.svg')
plt.show()

#%% Save fitted results
fitted_results_df = pd.DataFrame(fitted_results)
fitted_results_df.to_csv(f'{output_prefix}_fitted_results.csv', index=False)