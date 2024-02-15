import numpy as np
def garch_mle(params, returns,realized_vol):
    omega = params[0]
    alpha = params[1]
    beta = params[2]


    conditional = np.zeros(len(returns))
    conditional[0] = realized_vol[0]
    for t in range(1, len(returns)):
        conditional[t] = np.sqrt(omega + alpha * returns[t - 1] ** 2 + beta * realized_vol[t - 1] ** 2)

    likelihood = 1/((2*np.pi)**(1/2)*conditional)*np.exp(-returns**2/(2*conditional**2))
    log_likelihood = np.sum(np.log(likelihood))
    return -log_likelihood


