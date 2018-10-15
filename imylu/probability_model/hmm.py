# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-09-27 14:09:22
@Last Modified by:   tushushu
@Last Modified time: 2018-09-27 14:09:22
"""

from collections import defaultdict
from itertools import chain


class HMM(object):
    """HMM class.

    Attributes:
        states {set} -- Distinct states appeared in the train set.
        observations {set} -- Distinct observations appeared in the train set.
        start_prob {dict} -- Start probability = P(States_i)
        trans_prob {dict} -- Transition probability = P(States_i → States_j)
        emit_prob {dict} -- Emission probability = P(States_i → Observation_j)
    """

    def __init__(self):
        self.states = None
        self.observations = None
        self.start_probs = None
        self.trans_probs = None
        self.emit_probs = None

    def _get_prob(self, element_cnt):
        """Convert values into log probabilities in the dictionary.

        Arguments:
            element_cnt {defaultdict} -- Count of elements.

        Returns:
            dict -- Log probabilities of elements.
        """

        ret = defaultdict(float)
        n_elements = sum(element_cnt.values())
        for element, cnt in element_cnt.items():
            ret[element] = cnt / n_elements
        return ret

    def fit(self, X, y):
        """Calculate start probability, transition probability and emission probability
        by Maximum likelihood estimation.

        Likelihood function:
        L = p^sum(x|p) * (1-p)^sum(x|(1-p))

        Take the logarithm of both sides of this equation:
        log(L) = log(p) * sum(x|p) + log(1-p) * sum(x|(1-p))

        Get p derivative:
        1/p * sum(x|p) - 1/(1-p) * sum(x|(1-p)) = 0
        (1-p) * sum(x|p) - p * sum(x|(1-p) = 0
        p = sum(x|p) / sum(x)
        --------------------------------------------------------------------------------

        Arguments:
            X {list} -- 2d list with observations.
            y {list} -- 2d list with states.

        Returns:
            start_prob {dict} -- {'state_1': p, 'state_2': p...}
            trans_prob {dict} -- {'state_1': {'state_1': p, 'state_2': p...},
            'state_2': {'state_1': p, 'state_2': p...}...}
            emit_prob {dict} -- {'state_1': {'observation_1': p...
            'observation_n': p}...'state_n': {observation_1': p...
            'observation_n': p}}
        """
        # Use defaultdict to avoid the case when key is not in the dictionary.
        start_cnt = defaultdict(int)
        trans_cnt = defaultdict(lambda: defaultdict(int))
        emit_cnt = defaultdict(lambda: defaultdict(int))
        # Count the number of occurrences.
        for observations, states in zip(X, y):
            n = len(states)
            for i in range(n - 1):
                start_cnt[states[i]] += 1
                trans_cnt[states[i]][states[i + 1]] += 1
                emit_cnt[states[i]][observations[i]] += 1
            start_cnt[states[n - 1]] += 1
            emit_cnt[states[n - 1]][observations[n - 1]] += 1
        # Convert values into probabilities in the dictionary.
        self.start_probs = self._get_prob(start_cnt)
        self.trans_probs = {k: self._get_prob(v) for k, v in trans_cnt.items()}
        self.emit_probs = {k: self._get_prob(v) for k, v in emit_cnt.items()}
        # Get unique states and observations.
        self.states = set(self.start_probs.keys())
        self.observations = set(chain(*X))

    def get_emit_prob(self, state, observation):
        """Calculate emission  probability, when the observation has not
        appearred in the train data, we set emission  probability to 1.

        Arguments:
            state {str}
            observation {str}
        """

        if observation in self.observations:
            ret = self.emit_probs[state][observation]
        else:
            ret = 1
        return ret

    def _viterbi(self, observations):
        """Viterbi algorithm.

        Arguments:
            observations {list} -- observation_1...observation_n-1

        Returns:
            tuple -- states, probabilities
        """

        # Initialization.
        observations = iter(observations)
        observation = next(observations)
        # Record the sequence of states.
        paths = [[state] for state in self.states]
        # Joint probability.
        probs = [self.start_probs[state] * self.get_emit_prob(
            state, observation) for state in self.states]
        # Iterate the observations left.
        for observation in observations:
            new_paths = []
            new_probs = []
            for state_cur in self.states:
                prob_max = -float('inf')
                m = len(paths)
                # Calculate the best path to current state.
                for i, path, prob in zip(range(m), paths, probs):
                    state_pre = path[-1]
                    # Multiply the log prob of current state.
                    trans_prob = self.trans_probs[state_pre][state_cur]
                    emit_prob = self.get_emit_prob(state_cur, observation)
                    prob_mul = prob * trans_prob * emit_prob
                    # Choose the maximum sum result.
                    if prob_mul > prob_max:
                        prob_max = prob_mul
                        idx_best = i
                new_paths.append(paths[idx_best] + [state_cur])
                new_probs.append(prob_max)
            # Update paths, probs
            paths = new_paths
            probs = new_probs
        return paths, probs

    def _predict(self, Xi):
        """Predict the states according to observations.

        Arguments:
            Xi {list} -- Observations

        Returns:
            list -- States
        """

        ret = zip(*self._viterbi(Xi))
        return max(ret, key=lambda x: x[1])[0]

    def predict(self, X):
        """Predict the states according to observations.

        Arguments:
            X {list} -- 2D list with observations.

        Returns:
            list -- 2D list with states.
        """

        return [self._predict(Xi) for Xi in X]
