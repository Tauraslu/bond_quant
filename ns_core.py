import numpy as np

def nelson_siegel(tau, beta0, beta1, beta2, lambd):
    term1 = beta0
    term2 = beta1 * (1 - np.exp(-lambd * tau)) / (lambd * tau)
    term3 = beta2 * ((1 - np.exp(-lambd * tau)) / (lambd * tau) - np.exp(-lambd * tau))
    return term1 + term2 + term3

def ns_yield(t, beta0, beta1, beta2, lambd):
    term1 = (1 - np.exp(-t / lambd)) / (t / lambd)
    term2 = term1 - np.exp(-t / lambd)
    return beta0 + beta1 * term1 + beta2 * term2
