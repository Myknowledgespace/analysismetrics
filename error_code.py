### Developed by YPK, UPC Barcelona####
import numpy as np

# P= Predicted values (vector)
# T= Target values  (vector)
def error(P, T):
    n = P.size

    # Ensure P and T have the same shape
    if P.shape != T.shape:
        raise ValueError("P and T must have the same shape.")

    # Calculate SSE (Sum of square of errors)
    SSE = np.sum((P - T) ** 2)

    # Calculate RMSE (Root mean square error)
    RMSE = np.sqrt(SSE / n)

    # Calculate standard deviations
    StdT = np.std(T, axis=0, ddof=0)  # Population std deviation
    StdP = np.std(P, axis=0, ddof=0)

    # Calculate NRMSE (normalized root mean square error)
    NRMSE = 100 * RMSE / StdT

    # Calculate NSC (Nash sutcliffe efficiency)
    SS_total = np.sum((T - np.mean(T)) ** 2)
    NSC = 1 - (SSE / SS_total)

    # Calculate correlation coefficient
    Cor = np.sum((P - np.mean(P)) * (T - np.mean(T))) / (np.sqrt(np.sum((P - np.mean(P)) ** 2)) * np.sqrt(np.sum((T - np.mean(T)) ** 2)))

    # Calculate MAE (mean absolute value of errors)
    MAE = np.sum(np.abs(P - T)) / n

    # Calculate MARE (Mean absolute relative errors)
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_errors = np.abs((T - P) / T)
        MARE = np.sum(relative_errors) / n

    # Calculate PERS
    if P.ndim == 2 and T.ndim == 2 and P.shape[1] > 1:
        P2 = T[:, :-1]
        T2 = T[1:, :]
        SSEN = np.sum((P2 - T2) ** 2)
        PERS = 1 - (SSE / SSEN)
        RMSEN = np.sqrt(SSEN / (n - 1))
        NRMSEN = 100 * RMSEN / np.std(T2, axis=0, ddof=0)
    else:
        PERS = None
        RMSEN = None
        NRMSEN = None

    # Error
    Er = P - T

    # Return the calculated metrics
    return {
        'RMSE': RMSE,
        'Cor': Cor,
        'NSC': NSC,
        'MAE': MAE,
        'STD_T': StdT,
        'STD_P': StdP
    }
