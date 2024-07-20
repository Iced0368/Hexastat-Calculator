import numpy as np
from collections import defaultdict

class MarkovMatrix:
    def __init__(self):
        self.transitions = {}
        self.states = []
        self.state_index = {}
        self.compiled_matrix = None

    def add_transition(self, from_state, to_state, prob):
        if prob == 0:
            del self.transitions[(from_state, to_state)]
        else:
            self.transitions[(from_state, to_state)] = prob

    def clear_transition(self, from_state):
        fsstate_to_remove = []
        for fsstate in self.transitions.keys():
            fstate, to_state = fsstate
            if fstate == from_state:
                fsstate_to_remove.append(fsstate)

        for fsstate in fsstate_to_remove:
            del self.transitions[fsstate]

    def compile(self):
        # Create a unique list of all states
        states = set()
        for fsstate in self.transitions.keys():
            from_state, to_state = fsstate
            states.add(from_state)
            states.add(to_state)

        self.states = list(states)
        # Create a mapping from state to index
        self.state_index = {state: i for i, state in enumerate(self.states)}

        # Initialize the transition matrix
        size = len(self.states)
        self.compiled_matrix = np.zeros((size, size))

        # Fill the transition matrix with probabilities
        for fsstate, prob in self.transitions.items():
            from_state, to_state = fsstate

            from_index = self.state_index[from_state]
            to_index = self.state_index[to_state]

            self.compiled_matrix[from_index][to_index] = prob


    def get_matrix(self):
        if self.compiled_matrix is None:
            raise ValueError("The matrix has not been compiled. Call compile() first.")
        return self.compiled_matrix

    def get_states(self):
        return self.states

    def get_state_index(self):
        return self.state_index

    def create_vector(self, initial_states):
        """
        initial_states can be a single state (str or tuple) or a dictionary of states with their assigned probabilities.
        """
        vector = np.zeros(len(self.states))
        
        if isinstance(initial_states, (str, tuple)):
            if initial_states not in self.state_index:
                raise ValueError(f"State '{initial_states}' not found in state index.")
            vector[self.state_index[initial_states]] = 1.0
        elif isinstance(initial_states, dict):
            for state, prob in initial_states.items():
                if state not in self.state_index:
                    raise ValueError(f"State '{state}' not found in state index.")
                vector[self.state_index[state]] = prob
        else:
            raise ValueError("initial_states should be either a single state (str or tuple) or a dictionary of states with probabilities.")
        
        return vector

    def vector_to_state_dict(self, vector):
        """
        Converts a probability vector back to a dictionary of states with their probabilities.
        """
        if len(vector) != len(self.states):
            raise ValueError("The length of the vector does not match the number of states.")
        state_probabilities = {self.states[i]: vector[i] for i in range(len(self.states))}
        return state_probabilities
    
    def compress_state_dict(self, state_dict, indices):
        """
        Compresses the state dictionary based on the specified indices.
        """
        compressed = defaultdict(float)
        for state, value in state_dict.items():
            if isinstance(state, tuple):
                compressed_key = tuple(state[i] for i in indices)
            else:
                compressed_key = (state,)
            compressed[compressed_key] += value
        return dict(compressed)