import numpy as np

INF = 1 << 64

def reinforce(table, x, num_reinforcements, do_compress=True):
    init = table.create_vector(x)
    result = init @ np.linalg.matrix_power(
        table.get_matrix(),
        num_reinforcements
    )
    if do_compress:
        return table.compress_state_dict(table.vector_to_state_dict(result), [0])
    else:
        return result


def steady_state_distribution(M):
    pi = np.linalg.matrix_power(M, INF)
    return pi


def matrix_semi_inf_geometric_series(M):
    I = np.eye(M.shape[0])

    Q = steady_state_distribution(M)
    Z = M - Q

    inv_matrix = np.linalg.inv(I - Z)
    
    return inv_matrix 


def matrix_power_series(M, n):
    I = np.eye(M.shape[0])

    Q = steady_state_distribution(M)
    Z = M - Q

    Z_powered = np.linalg.matrix_power(Z, n+1)
    inv_matrix = np.linalg.inv(I - Z)
    
    return (I - Z_powered) @ inv_matrix + n*Q
    