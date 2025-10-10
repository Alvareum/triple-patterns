from scipy import special, stats
import numpy as np
import numpy.typing as npt
from typing import Tuple


# Map from names to lambda_ values used in power_divergence().
_power_div_lambda_names = {
    "pearson": 1,
    "log-likelihood": 0,
    "freeman-tukey": -0.5,
    "mod-log-likelihood": -1,
    "neyman": -2,
    "cressie-read": 2/3,
}
def _count(a:npt.ArrayLike, axis:int=None)->int:
    """Count the number of non-masked elements of an array.

    This function behaves like `np.ma.count`, but is much faster
    for ndarrays.
    """
    if hasattr(a, 'count'):
        num = a.count(axis=axis)
        if isinstance(num, np.ndarray) and num.ndim == 0:
            # In some cases, the `count` method returns a scalar array (e.g.
            # np.array(3)), but we want a plain integer.
            num = int(num)
    else:
        if axis is None:
            num = a.size
        else:
            num = np.prod(np.array(a.shape)[np.array(axis)])
    return num

def power_divergence_vectorized(f_obs:npt.ArrayLike, f_exp:npt.ArrayLike=None, ddof:int=0, axis:Tuple[int, int]=(1,2), lambda_:str=None) -> Tuple[float, float]:
    """
    Copy of scipy.stats with small modifications for vectorization.

    Vectorization was done by Krivonosov Mikhail / Lobachevsky University.
    """

    if isinstance(lambda_, str):
        if lambda_ not in _power_div_lambda_names:
            names = repr(list(_power_div_lambda_names.keys()))[1:-1]
            raise ValueError(f"invalid string for lambda_: {lambda_!r}. "
                             f"Valid strings are {names}")
        lambda_ = _power_div_lambda_names[lambda_]
    elif lambda_ is None:
        lambda_ = 1

    f_obs = np.asanyarray(f_obs)
    f_obs_float = f_obs.astype(np.float64)
    if f_exp is not None:
        f_exp = np.asanyarray(f_exp)
        rtol = 1e-8  # to pass existing tests
        with np.errstate(invalid='ignore'):
            f_obs_sum = f_obs_float.sum(axis=axis)
            f_exp_sum = f_exp.sum(axis=axis)
            relative_diff = (np.abs(f_obs_sum - f_exp_sum) /
                             np.minimum(f_obs_sum, f_exp_sum))
            diff_gt_tol = (relative_diff > rtol).any()
        if diff_gt_tol:
            msg = (f"For each axis slice, the sum of the observed "
                   f"frequencies must agree with the sum of the "
                   f"expected frequencies to a relative tolerance "
                   f"of {rtol}, but the percent differences are:\n"
                   f"{relative_diff}")
            raise ValueError(msg)
    else:
        # Ignore 'invalid' errors so the edge case of a data set with length 0
        # is handled without spurious warnings.
        with np.errstate(invalid='ignore'):
            f_exp = f_obs.mean(axis=axis, keepdims=True)   

    if lambda_ == 1:
        # Pearson's chi-squared statistic
        terms = (f_obs_float - f_exp)**2 / f_exp
    elif lambda_ == 0:
        # Log-likelihood ratio (i.e. G-test)
        terms = 2.0 * special.xlogy(f_obs, f_obs / f_exp)
    elif lambda_ == -1:
        # Modified log-likelihood ratio
        terms = 2.0 * special.xlogy(f_exp, f_exp / f_obs)
    else:
        # General Cressie-Read power divergence.
        terms = f_obs * ((f_obs / f_exp)**lambda_ - 1)
        terms /= 0.5 * lambda_ * (lambda_ + 1)
    stat = terms.sum(axis=axis)
    num_obs = _count(terms, axis=axis)
    ddof = np.asarray(ddof)
    p = stats.chi2.sf(stat, num_obs - 1 - ddof)
    return stat, p


def chi2_contingency_vectorized(contingency_table:npt.ArrayLike, lambda_:str='pearson')-> Tuple[float, float, npt.ArrayLike]:
    """
    Implementation and explanation of simple chi2 contigency got from 
    https://habr.com/ru/companies/mygames/articles/677074/
    
    einsum explanation:
    https://habr.com/ru/articles/544498/

    Vectorization was done by Krivonosov Mikhail / Lobachevsky University.

    Parameters:
        contingency_table : (n, m, k) array of n contigency tables with same shape (m, k).
    
    Returns:
        p-values : array of n elements - p-values for each of n contigency tables. 
    """
     
    # total number of observations
    total = contingency_table.sum(axis=(1, 2)).reshape(-1, 1)
    
    # sum by rows and columns
    col_probs = contingency_table.sum(axis = 1)
    row_probs = contingency_table.sum(axis = 2)

    # convert to probabilities
    col_probs = col_probs / total
    row_probs = row_probs / total
    
    # compute expected values
    col_probs = col_probs.reshape(col_probs.shape[0], 1, col_probs.shape[1])
    row_probs = row_probs.reshape(row_probs.shape[0], row_probs.shape[1], 1)
    expected = np.einsum("ijk, iks->ijs", row_probs, col_probs)
    
    expected = expected * total.reshape(-1, 1, 1)
    
    # compute degrees of freedom
    dof = (contingency_table.shape[1] - 1) * (contingency_table.shape[2] - 1)
    
    chi2, p = power_divergence_vectorized(contingency_table, expected,
                               ddof=contingency_table[0].size - 1 - dof, axis=(1,2),
                               lambda_=lambda_)
    return p, chi2, expected