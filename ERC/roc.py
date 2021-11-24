import  numpy as np
import operator

def calculate_roc(gscores, iscores, ds_scores=False, rates=True):
    """Calculates FMR, FNMR
    @param gscores: Genuine matching scores
    @type gscores: Union[list, ndarray]
    @param iscores: Impostor matching scores
    @type giscores: Union[list, ndarray]
    @param ds_scores: Indicates whether input scores are
        dissimilarity scores
    @type ds_scores: bool
    @param rates: Indicates whether to return error rates instead
        of error values
    @type rates: bool
    @return: (thresholds, FMR, FNMR) or (thresholds, FM, FNM)
    @rtype: tuple
    """
    if isinstance(gscores, list):
        gscores = np.array(gscores, dtype=np.float64)

    if isinstance(iscores, list):
        iscores = np.array(iscores, dtype=np.float64)

    if gscores.dtype == np.int:
        gscores = np.float64(gscores)

    if iscores.dtype == np.int:
        iscores = np.float64(iscores)

    if ds_scores:
        gscores = gscores * -1
        iscores = iscores * -1

    gscores_number = len(gscores)
    iscores_number = len(iscores)

    # Labeling genuine scores as 1 and impostor scores as 0
    gscores = zip(gscores, [1] * gscores_number)
    iscores = zip(iscores, [0] * iscores_number)

    # Python3 compatibility
    gscores = list(gscores)
    iscores = list(iscores)

    # Stacking scores
    scores = np.array(sorted(gscores + iscores, key=operator.itemgetter(0)))
    cumul = np.cumsum(scores[:, 1])

    # Grouping scores
    thresholds, u_indices = np.unique(scores[:, 0], return_index=True)

    # Calculating FNM and FM distributions
    fnm = cumul[u_indices] - scores[u_indices][:, 1]  # rejecting s < t
    fm = iscores_number - (u_indices - fnm)

    # Calculating FMR and FNMR
    if rates:
        fnm_rates = fnm / gscores_number
        fm_rates = fm / iscores_number
    else:
        fnm_rates = fnm
        fm_rates = fm

    if ds_scores:
        return thresholds * -1, fm_rates, fnm_rates

    return thresholds, fm_rates, fnm_rates

def get_fmr_op(fmr, fnmr, op):
    """Returns the value of the given FMR operating point
    Definition:
    ZeroFMR: is defined as the lowest FNMR at which no false matches occur.
    Others FMR operating points are defined in a similar way.
    @param fmr: False Match Rates
    @type fmr: ndarray
    @param fnmr: False Non-Match Rates
    @type fnmr: ndarray
    @param op: Operating point
    @type op: float
    @returns: Index, The lowest FNMR at which the probability of FMR == op
    @rtype: float
    """
    index = np.argmin(abs(fmr - op))
    return index, fnmr[index]

def get_fnmr_op(fmr, fnmr, op):
    """Returns the value of the given FNMR operating point
    Definition:
    ZeroFNMR: is defined as the lowest FMR at which no non-false matches occur.
    Others FNMR operating points are defined in a similar way.
    @param fmr: False Match Rates
    @type fmr: ndarray
    @param fnmr: False Non-Match Rates
    @type fnmr: ndarray
    @param op: Operating point
    @type op: float
    @returns: Index, The lowest FMR at which the probability of FNMR == op
    @rtype: float
    """
    temp = abs(fnmr - op)
    min_val = np.min(temp)
    index = np.where(temp == min_val)[0][-1]
    #index = np.argmin(abs(fnmr - op))

    return index, fmr[index]

def get_eer_threshold(gen_scores, imp_scores, hformat=False, ds_scores=False):
    """Calculates EER associated statistics
    Keyword Arguments:
    @param gen_scores: The genuine scores
    @type gen_scores: list
    @param imp_scores: The impostor scores
    @type imp_scores: list
    @param id: An id for the experiment
    @type id: str
    @param hformat: Indicates whether the impostor scores are in histogram
        format
    @type hformat: bool
    @param ds_scores: Indicates whether the input scores are dissimilarity
        scores
    @type ds_scores: bool
    """

    # Calculating probabilities using scores as thrs
    roc_info = calculate_roc(gen_scores, imp_scores,
                                 ds_scores, rates=False)
    gnumber = len(gen_scores)
    inumber = len(imp_scores)
    thrs, fm, fnm = roc_info
    fmr = fm / inumber
    fnmr = fnm / gnumber
    ind, fmr1000 = get_fmr_op(fmr, fnmr, 0.001)
    fmr1000_th = thrs[ind]

    ind, fmr100 = get_fmr_op(fmr, fnmr, 0.01)
    fmr100_th = thrs[ind]

    ind, fmr10000 = get_fmr_op(fmr, fnmr, 0.0001)
    fmr10000_th = thrs[ind]







    return fmr100_th, fmr1000_th, fmr10000_th
