import numpy as np
def garch_mle(params, returns):
    omega = params[0]
    alpha = params[1]
    beta = params[2]

    long_run = (omega / (1 - alpha - beta)) ** (1 / 2)
    resid = returns
    realised = abs(resid)

    conditional = np.zeros(len(returns))
    conditional[0] = long_run
    for t in range(1, len(returns)):
        conditional[t] = (omega + alpha * resid[t - 1] ** 2 + beta * conditional[t - 1] ** 2) ** (1 / 2)

    likelihood = 1/((2*np.pi)**(1/2)*conditional)*np.exp(-realised**2/(2*conditional**2))
    log_likelihood = np.sum(np.log(likelihood))
    return -log_likelihood