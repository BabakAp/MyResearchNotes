# Babak Alipour (babak.alipour@gmail.com, babak.ap@ufl.edu)
# Last edit: 03242019
import sys
print('Python v' + sys.version)
import pandas as pd
import numpy as np
from numba import jit
import numba
import functools
import os, math, re, warnings, gc, pickle, gzip, io, time
import matplotlib as mpl
import matplotlib.pyplot as plt
from importlib import reload
import itertools
# inline_rc = dict(mpl.rcParams)
import scipy
from scipy import stats as st, integrate
import statsmodels as sm
import seaborn as sns
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm, tnrange, tqdm_notebook
from scipy.stats import ttest_ind, ttest_rel, mannwhitneyu, ranksums, wilcoxon, kstest, shapiro, ks_2samp
from scipy.spatial import distance
random_state = 1000  #seed for sampling
np.random.seed(random_state)
from sklearn import preprocessing, metrics
from sklearn.model_selection import KFold, train_test_split, cross_val_score
import psutil
from datetime import datetime
from collections import Counter
from collections import defaultdict
# import blist # Faster asymptotic list (tree-based).
current_ms = lambda: int(round(time.time() * 1000))
unit = lambda x: x
KB = 1024; MB = KB*1024; GB = MB*1024
K = 1000; M = K*1000; B = M*1000
# Plotting params
a4_dims = (11.7, 8.27)
wide_dims = (16, 9)
# Useful beeping function.
import platform

if (platform.system() == "Windows"):
    import winsound

    def beep():
        frequency = 2500  # Set Frequency in Hz.
        duration = 500  # Set Duration in ms.
        winsound.Beep(frequency, duration)

# Retrieve name of a variable as a str.
import inspect


def retrieve_name(var):
        """
        Gets the name of var. Does it from the out most frame inner-wards.
        :param var: variable to get name from.
        :return: string
        """
        for fi in reversed(inspect.stack()):
            names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
            if len(names) > 0:
                return names[0]

# Print large integer with ',' every thousand e.g. 67510193 -> '67,510,193'.
def prettyInteger(integer):
    return "{:,}".format(integer)
def prettyInt(integer):
    return prettyInteger(integer)
def prettyPrintInteger(integer):
    print(prettyInteger(integer))
def prettyPrintInt(integer):
    print(prettyInteger(integer))	

# Valid line styles: '-' | '--' | '-.' | ':' ,'steps' 
def cdf_plot(ser,ax=None,figsize=(7,5), label=None, fontSize = 15, lineWidth=2, lineStyle='-', ylabel='CDF'):
    print(len(ser))
    ser = ser.sort_values()
    cum_dist = np.linspace(0.,1.,len(ser))
    ser_cdf = pd.Series(cum_dist, index=ser)
    ax = ser_cdf.plot(drawstyle='steps',figsize=figsize,yticks=np.arange(0.,1.001,0.1),ax=ax, label=label, linewidth=lineWidth, linestyle=lineStyle)
    ## Change x axis font size
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontSize)

    ## Change y axis font size
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontSize)
        
    ax.set_ylabel(ylabel, fontsize=18)    
    return ax
# Other helper functions
def box_plot(df):
    return df.boxplot(return_type='axes',showfliers=False,showmeans=True, whis='range')
# http://stackoverflow.com/a/39424972
# Using InterQuartileRange(IQR) to reject outliers
def reject_outliers(ser, iq_range=0.9):
    if (ser.empty):
        return ser
    q = (1 - iq_range)/2
    qlow, median, qhigh = ser.dropna().quantile([q, 0.50, 1-q])
    iqr = qhigh - qlow
    len_orig = len(ser)
    ser = ser[ (ser - median).abs() <= iqr]
#     left = qlow - 1.5*iqr
#     right = qhigh + 1.5*iqr
#     ser = ser[ ser<= right]
#     ser = ser[ ser>= left]
    print( str(len_orig-len(ser)) + " removed! That is " + str(int((len_orig-len(ser))/len_orig*100000)/1000) + "% of original")
    return ser
def addBoxes(s, l, ax):
    ## small textbox containing these params will be added
    l_median = l.median()
    if (int(l_median) == l_median):
        pc_txt = 'l\n$\mu=%.2f$\n$\mathrm{median}=%d$\n$\sigma=%.2f$'%(l.mean(), int(l_median), l.std())
    else:
        pc_txt = 'l\n$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$'%(l.mean(), l.median(), l.std())
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ## small textbox containing these params will be added
    s_median = s.median()
    if (int(s_median) == s_median):
        ph_txt = 's\n$\mu=%.2f$\n$\mathrm{median}=%d$\n$\sigma=%.2f$'%(s.mean(), int(s_median), s.std())
    else:
        ph_txt = 's\n$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$'%(s.mean(), s.median(), s.std())
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.35, 0.35, pc_txt, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    ax.text(0.35, 0.85, ph_txt, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)

def stat_tests(s, l):
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
    #Using Welch's t-test (unpaired or independent t-test) https://en.wikipedia.org/wiki/Welch's_t-test
    t, p = ttest_ind(s.sample(frac=0.01, replace=False, random_state=random_state), 
                     l.sample(frac=0.01, replace=False, random_state=random_state),
                     equal_var=False)
    print(t)
    print("s, l -- p-value (Welch's t-test): " + str(p))
    t, p = mannwhitneyu(s.sample(frac=0.01, replace=False, random_state=random_state), 
                 l.sample(frac=0.01, replace=False, random_state=random_state), alternative='two-sided')
    print("s, l -- p-value (mannwhitney's u-test): " + str(p))

def stat_tests_large(s, l):
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
    #Using Welch's t-test (unpaired or independent t-test) https://en.wikipedia.org/wiki/Welch's_t-test
    t, p = ttest_ind(s, l, equal_var=False)
    print(t)
    print("s, l -- p-value (Welch's t-test): " + str(p))
    t, p = mannwhitneyu(s, l, alternative='greater')
    print("s, l -- p-value (mannwhitney's u-test): " + str(p))

distributions = [st.expon, st.norm,st.gamma,st.weibull_max,st.weibull_min,st.logistic,st.beta, st.lognorm]
def bestFit(data):
    mles = []
    for distribution in distributions:
        pars = distribution.fit(data)
        mle = distribution.nnlf(pars, data)
        mles.append(mle)
    temp = sorted(zip(distributions, mles), key=lambda d: d[1])
    print(temp[0][0].name,temp[1][0].name,temp[2][0].name,temp[3][0].name,
          temp[4][0].name, temp[5][0].name, temp[6][0].name, temp[7][0].name)
    print(temp[0][1],temp[1][1],temp[2][1],temp[3][1],temp[4][1], temp[5][1], temp[6][1], temp[7][1])
sns.reset_orig()
###### FOR HEATMAP, PROFILECAST AND SPIRIT ANALYSIS
# Using pandas built-in crosstab instead of groupby and manual loop results in ~15-20x speedup
def getMatrixFast(label, df, field1, field2, heat, fn=np.log, aggfunc=np.sum):
    start = current_ms()
    print("Creating the matrix for label: '{}' heat: '{}'...".format(label,heat))
    data = pd.crosstab(index=df[field1],columns=df[field2],values=df[heat], aggfunc=aggfunc)
    if (fn is np.log):
        data = data.fillna(value=1.0).apply(np.log).as_matrix() #fill NaN with 1, so that the log takes it to 0, get the array
    else:
        data = data.fillna(value=0.0).apply(fn).as_matrix()
    data = preprocessing.normalize(data,norm='l1',copy=False) #row normalize using L1 norm
    print("Done in " + str((current_ms()-start)/1000) + "s")
    return data
# Using pandas built-in crosstab instead of groupby and manual loop results in ~15-20x speedup
def getDFFast(label, df, field1, field2, heat, fn=np.log):
    start = current_ms()
    print("Creating the matrix for label: '{}' heat: '{}'...".format(label,heat))
    data = pd.crosstab(index=df[field1],columns=df[field2],values=df[heat], aggfunc=np.sum)
    if (fn is np.log):
        data = data.fillna(value=1.0).apply(np.log) #fill NaN with 1, so that the log takes it to 0, get the array
    else:
        data = data.fillna(value=0.0).apply(fn)
    data = data.div(data.sum(axis=1), axis=0) #row normalize using L1 norm
    print("Done in " + str((current_ms()-start)/1000) + "s")
    return data

# Generates data (unless provided) and draws heatmap
def heatmap(label, df, field1, field2, heat, data=None, save=True, saveAppendix=''):
    if (data is None):
        data = getMatrixFast(label, df, field1, field2, heat)
    start = current_ms()
    print("Plotting heatmap...")
    fig, ax = plt.subplots();
    # Create the heatmap.
    ax.pcolormesh(data, cmap=plt.cm.hot)
    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
    cats = df[field2].unique().tolist(); cats.sort()
    ax.set_xlim(0,len(cats))
    users = df[field1].unique().tolist(); users.sort()
    ax.set_ylim(0,len(users))
    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.set_xticklabels(cats, minor=False)
    if (len(users) < 200):
        ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)
        ax.set_yticklabels(users, minor=False)
    fig.set_size_inches(18.5, 10.5,forward=True)
    filename=label + '_' + heat + '_' + field1 + '_' + field2 + '_normalized_' + saveAppendix + '.png'
    if (save): 
        fig.savefig(filename, dpi=160)
        print("saved file ", filename)
    print("Done in " + str((current_ms()-start)/1000) + "s")
    plt.show()


def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size/2:]
# Cross-Validation
def cv(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=5)
    print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

###### Going from AP names to building prefix.
prefix_pattern = re.compile('(^[^\d^\W]*)')
## Hardcoded, because rl1/rl2, ohl1/ohl2, ff1
ap_loc_except = set(['rl','ohl','ff'])
def to_prefix(apname):
    ap_prefix = prefix_pattern.search(apname).group(1)
    if ap_prefix in ap_loc_except:
        return apname[:len(ap_prefix)+1]
    elif ap_prefix:
        return ap_prefix
    else:
        return "b" # Use 'b' since that's unknown.

###### Helper functions used in Encounter paper with Mimonah.
def getAverageOfList(l):
    return sum(l)/len(l)

#    -2.2e-16 is an adjustment to prevent math domain error on acos of numbers slightly >1 due to float precision error.
#    math.acos is faster than np.arccos for a single scalar.
#    @jit results in 3x speedup.
@jit
def getAngularSim(u1, u2):
    return 1 - ( math.acos( np.dot(u1, u2)/ (np.linalg.norm(u1, 2) * np.linalg.norm(u2, 2)) - 2.2e-16) / math.pi )
@jit
def getAngularSimFromCos(cos):
    return 1 - ( math.acos(cos - 2.2e-16) / math.pi )
# Returns cosine similiarity of two input vectors.
@jit
def getCosineSim(u1, u2):
    return np.dot(u1, u2) / ( np.linalg.norm(u1, 2) * np.linalg.norm(u2, 2) )

# Equivalent to 1-getCosineSim, added to my_import on 02222018.
# This is MUCH faster than nltk.cluster.util.cosine_distance
@jit
def getCosineDist(u1, u2):
    return 1 - ( np.dot(u1, u2) / ( np.linalg.norm(u1, 2) * np.linalg.norm(u2, 2) ) )

# If a dataframe has a field where each entry is an iterable,
# this function flattens all of those entries into *one* list.
def flatten(df, field):
    return list(itertools.chain.from_iterable(df[field]))

# From GMM/RBM notebooks, updated to return p-values as well.
def computeDValuesWithTestData(test_data, reconstructed_test_data, n_bins=100):
    d_values = []
    p_values = []
    jsds = []
    for index in range(
        0, test_data.shape[1]
    ):  # -1 is to remove the device type column.
        original = pd.Series(test_data[:, index])
        synthesized = pd.Series(reconstructed_test_data[:, index])
        # Compute ks stat.
        d, p = ks_2samp(original, synthesized)
        # Create histograms (probabilities)
        orig_probs, _ = np.histogram(a=original, bins=n_bins)
        synth_probs, _ = np.histogram(a=synthesized, bins=n_bins)
        jsd = distance.jensenshannon(
            orig_probs, synth_probs, base=2.0
        )  # Default base is e.
        # Keep track of d-value, p-value and jsd.
        d_values.append(d)
        p_values.append(p)
        jsds.append(jsd)
    ###### return d-value stats:
    return d_values, p_values, jsds

def MyRound(arr, decimals = 3):
    return np.round(arr, decimals=decimals)

# Returns 5-number summary and more.
# (min, lower quartile, median, upper quartile, max, mean, std, skewness, kurtosis, gmean, hmean, sum)
# Expect a Pandas series or numpy array as input.
# Add 1e-6 to gmean, to prevent log(0) and, heam, to prevent division by zero.
# About kurtosis and Fisher vs Pearson definitions: https://en.wikipedia.org/wiki/Kurtosis
# "For the reason of clarity and generality, however, this article follows the non-excess convention and explicitly indicates where excess kurtosis is meant."
# There is only a -3 difference.
def getNumberSummary(series):
    # Handle the case where input is empty, then just return all 0s.
    if len(series) == 0:
        series = pd.Series([0, 0])
    return pd.DataFrame(
        {
            "min": [np.min(series)],
            "lower_quartile": [np.percentile(series, 25)],
            "median": [np.median(series)],
            "upper_quartile": [np.percentile(series, 75)],
            "max": [np.max(series)],
            "mean": [np.mean(series)],
            "std": [np.std(series, ddof=1 if len(series) > 1 else 0)],
            "skewness": [st.skew(series)],
            "kurtosis": [st.kurtosis(series, fisher=False)],
            "gmean": [st.gmean(series + 1e-6)],
            "hmean": [st.hmean(series + 1e-6)],
            "sum": [np.sum(series)]
        }
    )
