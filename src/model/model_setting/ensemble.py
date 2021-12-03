from copy import deepcopy
from itertools import combinations

def get_peaks(residual, period):
    pivot = 0
    peaks = {}
    while True:
        if pivot + period > len(residual):
            peak_value = max(residual[pivot:])
            peak_idx = residual[pivot:].index(peak_value)
            peaks[pivot + peak_idx] = peak_value
            break
        else:
            peak_value = max(residual[pivot:pivot + period])
            peak_idx = residual[pivot:pivot + period].index(peak_value)
            peaks[pivot + peak_idx] = peak_value
            pivot += period
    return peaks


# Return Dict of {index: scores} where index represents the location of anomaly within the time series
def ensemble(residuals, period, cut_off=0):
    candidates_scores = {}
    for residual in residuals:
        residual = residual[cut_off:]
        peaks = get_peaks(residual, period)
        peaks_sorted = sorted(list(peaks.keys()), key=lambda k: peaks[k], reverse=True)
        top = peaks_sorted[0]
        second = peaks_sorted[1]

        if top in candidates_scores:
            candidates_scores[cut_off + top] += peaks[top] / peaks[second]
        else:
            candidates_scores[cut_off + top] = peaks[top] / peaks[second]

        if second in candidates_scores:
            candidates_scores[cut_off + second] += peaks[second] / peaks[top]
        else:
            candidates_scores[cut_off + second] = peaks[second] / peaks[top]
    
    scores_clone = deepcopy(candidates_scores)
    for a, b in combinations(scores_clone.keys(), 2):
        if abs(a - b) < period:
            candidates_scores[a] += 0.5 * scores_clone[b]
            candidates_scores[b] += 0.5 * scores_clone[a]
    
    return candidates_scores
