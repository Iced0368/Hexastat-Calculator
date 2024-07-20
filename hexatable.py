import numpy as np
from markov import MarkovMatrix

hexa_prob = [
    np.array([0.35, 0.35, 0.35, 0.2, 0.2, 0.2, 0.2, 0.15, 0.1, 0.05, 0], dtype=np.float32),
    np.array([
        0.35, 0.35, 0.35, 0.2, 0.2, 
        0.2 *1.2, 0.2 *1.2, 0.15 *1.2, 0.1 *1.2, 0.05 *1.2, 
        0
    ], dtype=np.float32)
]

hex_frag_multiplier = 10
hexa_frag_cost = np.array([1, 1, 1, 2, 2, 2, 2, 3, 3, 5, 5])


def _build_hexa_table(prob):
    markov = MarkovMatrix()
    for i in range(11):
        for j in range(i, 20):
            if i+1 <= 10:
                markov.add_transition((i, j), (i+1, j+1), prob[i])
            markov.add_transition((i, j), (i, j+1), 1-prob[i])

    markov.compile()
    return markov

def _build_hexa_table_frag(prob, _hexa_frag_cost):
    markov = MarkovMatrix()
    for i in range(11):
        for j in range(i, 20):
            for k in range(_hexa_frag_cost[i]-1):
                markov.add_transition((i, j, k), (i, j, k+1), 1)

            if i+1 <= 10:
                markov.add_transition((i, j, _hexa_frag_cost[i]-1), (i+1, j+1, 0), prob[i])
            markov.add_transition((i, j, _hexa_frag_cost[i]-1), (i, j+1, 0), 1-prob[i])
    
    markov.compile()
    return markov

hexa_table = [
    _build_hexa_table(hexa_prob[0]),
    _build_hexa_table(hexa_prob[1])
]

hexa_table_frag = [
    _build_hexa_table_frag(hexa_prob[0], hexa_frag_cost),
    _build_hexa_table_frag(hexa_prob[1], hexa_frag_cost)
]