# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-09-27 14:09:22
@Last Modified by:   tushushu
@Last Modified time: 2018-09-27 14:09:22
https://blog.csdn.net/athemeroy/article/details/79339546
https://blog.csdn.net/athemeroy/article/details/79342048
"""

from collections import defaultdict


class HMM(object):
    """HMM class.

    Attributes:
        states {list}
        observations {list}
        start_prob {dict} -- start probability
        trans_prob {dict} -- transition probability
        emit_prob {dict} -- emission probability
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
            element_cnt {dict} -- Count of elements.

        Returns:
            dict -- Log probabilities of elements.
        """

        n_elements = sum(element_cnt.values)
        return {element: cnt / n_elements
                for element, cnt in element_cnt.items()}

    def fit(self, states_2d, observations_2d):
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
            states_2d {list} -- 2d list with states.
            observations_2d {list} -- 2d list with observations.

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
        for states, observations in zip(states_2d, observations_2d):
            n = len(states)
            for i in range(0, n - 1):
                start_cnt[states[i]] += 1
                trans_cnt[states[i]][states[i + 1]] += 1
                emit_cnt[states[i]][observations[i]] += 1
        # Convert values into probabilities in the dictionary.
        self.start_probs = self._get_prob(start_cnt)
        self.trans_probs = {k: self._get_prob(v) for k, v in trans_cnt.items()}
        self.emit_probs = {k: self._get_prob(v) for k, v in trans_cnt.items()}
        # Get unique states and observations.
        self.states = set(states)
        self.observations = set(observations)

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
        probs = [self.start_probs[state] * self.emit_probs[state]
                 [observation] for state in self.states]
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
                    emit_prob = self.emit_probs[state_cur][observation]
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

        ret = zip(self._viterbi(Xi))
        return max(ret, key=lambda x: x[1])[0]

    def predict(self, X):
        """Predict the states according to observations.

        Arguments:
            X {list} -- 2D list with observations.

        Returns:
            list -- 2D list with states.
        """

        return [self._predict(Xi) for Xi in X]


def test():
    model = HMM()
    model.start_probs = {"normal": 0.7, "light": 0.2, "heavy": 0.1}
    model.trans_probs = {"normal": {"normal": 0.7, "light": 0.2, "heavy": 0.1},
                         "light": {"normal": 0.4, "light": 0.4, "heavy": 0.2},
                         "heavy": {"normal": 0.2, "light": 0.5, "heavy": 0.3}}
    model.emit_probs = {"normal":
                        {"jump": 0.7, "cough": 0.1, "fever": 0, "shit": 0.2},
                        "light":
                        {"jump": 0.5, "cough": 0.2, "fever": 0.2, "shit": 0.1},
                        "heavy":
                        {"jump": 0.3, "cough": 0.2, "fever": 0.3, "shit": 0.2}
                        }
    model.states = {"normal", "light", "heavy"}
    model.observations = {"jump", "cough", "fever", "shit"}
    observations = ["fever", "fever", "fever"]
    print(model._viterbi(observations))


if __name__ == "__main__":
    test()
