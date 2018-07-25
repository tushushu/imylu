# -*- coding: utf-8 -*-
"""
@Author: tushushu 
@Date: 2018-07-17 18:04:28 
@Last Modified by: tushushu 
@Last Modified time: 2018-07-17 18:04:28 
"""
from collections import defaultdict


class Graph(object):
    """Graph class.
http://www.petrovi.de/data/universal.pdf
    Attributes:
        nodes {dict} -- A dict with Node names and Node indexes
        nexts {list} -- A list like [{Node indexes : {Node indexes : edge weights}}]
    """

    def __init__(self):
        self.nodes = {}
        self.nexts = []


class HMM(object):
    """HMM class.

    Attributes:
        start_prob {dict} -- start probability
        trans_prob {dict} -- transition probability
        emit_prob {dict} -- emission probability
    """

    def __init__(self, states, observations):
        self.start_prob = None
        self.trans_prob = None
        self.emit_prob = None

    def _MLE(self, states, observations):
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
            states {list} -- state_1, state_2...state_n
            observations {list} -- observation_1, observation_2...observation_n-1

        Returns:
            start_prob {dict} -- {'state_1': p, 'state_2': p...}
            trans_prob {dict} -- {'state_1': {'state_1': p, 'state_2': p...}, 
            'state_2': {'state_1': p, 'state_2': p...}...}
            emit_prob {dict} -- {'state_1': {'observation_1': p, 'observation_2': p...},
            'state_2': {'observation_1': p, 'observation_2': p...}...}
        """

        # Use defaultdict to avoid the case when key is not in the dict
        start_prob = defaultdict(int)
        trans_prob = defaultdict(lambda: defaultdict(int))
        emit_prob = defaultdict(lambda: defaultdict(int))
        # Count the number of occurrences
        n = len(states)
        for i in range(n):
            start_prob[states[i]] += 1
            trans_prob[states[i]][states[i+1]] += 1
            emit_prob[states[i]][observations[i]] += 1
        # Convert values into probabilities in the dictionary

        def prob(d):
            n = sum(d.values)
            return {k: v / n for k, v in d.items()}

        start_prob = prob(start_prob)
        trans_prob = {k: prob(v) for k, v in trans_prob.items()}
        emit_prob = {k: prob(v) for k, v in trans_prob.items()}

        return start_prob, trans_prob, emit_prob

    def _EM(self, states):
        raise NotImplementedError


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
