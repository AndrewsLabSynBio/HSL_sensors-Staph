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
plt.rcParams['text.usetex'] = False

plt.tick_params(axis='both', left=True, right=True, labelright=False,
                direction='out', length=2, width=2, colors='k', labelcolor='k')

params = {'axes.titlesize': 12, 'axes.labelsize': 12,
          'xtick.labelsize': 10, 'ytick.labelsize': 10}
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

#%% Hill functions
def hill(x, Kd, n, ymin, ymax):
    return ymin + (ymax - ymin) * (x**n) / (Kd**n + x**n)

def hill_res(params, ind_conc, exp_val):
    Kd, n, ymin, ymax = params
    return hill(ind_conc, Kd, n, ymin, ymax) - exp_val

#%% Read data
data = pd.read_csv(data_file)

plt.figure(figsize=(3, 3))

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

    regulator_results = {
        'Regulator': regulator,
        'Sensitivity_day1': None, 'Sensitivity_day2': None, 'Sensitivity_day3': None,
        'Kd_day1': None, 'Kd_day2': None, 'Kd_day3': None,
        'n_day1': None, 'n_day2': None, 'n_day3': None,
        'ymin_day1': None, 'ymin_day2': None, 'ymin_day3': None,
        'ymax_day1': None, 'ymax_day2': None, 'ymax_day3': None
    }

    #%% Per-day fits
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

        Kd, n, ymin, ymax = res_lsq.x
        day = day_col.split("_")[1]

        regulator_results[f'Kd_{day}'] = Kd
        regulator_results[f'n_{day}'] = n
        regulator_results[f'ymin_{day}'] = ymin
        regulator_results[f'ymax_{day}'] = ymax

        # Sensitivity at midpoint
        x_mid = Kd
        numerator = (ymax - ymin) * n * (Kd**n) * (x_mid**(n - 1))
        denominator = (Kd**n + x_mid**n)**2
        regulator_results[f'Sensitivity_{day}'] = numerator / denominator

    #%% Mean fit (plotting only)
    params0 = [1, 2, RPU_mean.min(), RPU_mean.max()]
    res_lsq_mean = least_squares(
        hill_res, params0, args=(inducer_conc, RPU_mean),
        bounds=([0, 0, 0.9 * RPU_mean.min(), 0.9 * RPU_mean.max()],
                [Kd_ubound, n_ubound, 1.2 * RPU_mean.min(), 1.2 * RPU_mean.max()])
    )

    Kd, n, ymin, ymax = res_lsq_mean.x

    color = subset['c_map'].iloc[0]

    plt.errorbar(
        inducer_conc, RPU_mean, yerr=sd_data,
        fmt='o', mfc='None', mec=color, ecolor=color,
        alpha=0.8, markersize=4, capsize=2, elinewidth=0.75
    )

    x_vals = np.logspace(
        np.log10(subset['HSL_concentraion'].min() * 0.9),
        np.log10(subset['HSL_concentraion'].max() * 4),
        1000
    )

    plt.plot(x_vals, hill(x_vals, *res_lsq_mean.x), lw=1, color=color)

    regulator_results.update({
        'Kd_mean': Kd,
        'n_mean': n,
        'ymin_mean': ymin,
        'ymax_mean': ymax
    })

    fitted_results.append(regulator_results)

#%% Axis formatting
plt.xscale('log')
plt.yscale('log')
plt.xlim(data['HSL_concentraion'].min() * 0.8,
         data['HSL_concentraion'].max() * 4)
plt.ylim([1e-3, 2e1])

plt.xlabel('pC-HSL (µM)')
plt.ylabel(r'$P_{Rpa}$ output (RPU)')

plt.gca().tick_params(which='major', direction='out', length=6, width=2)
plt.gca().tick_params(which='minor', direction='out', length=4, width=1)
plt.gca().xaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0))
plt.gca().xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs='auto'))
plt.gca().yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0))
plt.gca().yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs='auto'))

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{output_prefix}_combined.svg')
plt.show()

#%% Save results
pd.DataFrame(fitted_results).to_csv(
    f'{output_prefix}_fitted_results.csv', index=False
)