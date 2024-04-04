"""
This is the implementation of the Popularity-based Ranking-based Equal Opportunity (REO) metric.
It proceeds from a user-wise computation, and average the values over the users.
"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import operator
import scipy as sp

from elliot.evaluation.metrics.base_metric import BaseMetric


class PRU(BaseMetric):

    def __init__(self, recommendations, config, params, eval_objects):
        """
        Constructor
        :param recommendations: list of recommendations in the form {user: [(item1,value1),...]}
        :param config: SimpleNameSpace that represents the configuration of the experiment
        :param params: Parameters of the model
        :param eval_objects: list of objects that may be useful for the computation of the different metrics
        """
        super().__init__(recommendations, config, params, eval_objects)
        self._cutoff = self._evaluation_objects.cutoff
        self._relevance = self._evaluation_objects.relevance.binary_relevance
        self._pop_items = self._evaluation_objects.pop.get_pop_items()
        self._asc_sorted_pop_items = {item[0]: idx for idx, item in enumerate(sorted(self._pop_items.items(), key=operator.itemgetter(1), reverse=False))}
        pass


    @staticmethod
    def name():
        """
        Metric Name Getter
        :return: returns the public name of the metric
        """
        return "PRU"

    @staticmethod
    def __rec_popularity__(user_recommendations, pop_items):
        a = [pop_items[i] for i, _ in user_recommendations]
        if a.count(a[0]) == len(a):
            print(a)
            input(f"{user_recommendations}")
        return a

    def __pru(self, user_recommendations, cutoff):
        """
        Per User Recall
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param cutoff: numerical threshold to limit the recommendation list
        :param user_relevant_items: list of user relevant items in the form [item1,...]
        :return: the value of the Recall metric for the specific user
        """

        return -sp.stats.spearmanr(self.__rec_popularity__(user_recommendations[:cutoff], self._asc_sorted_pop_items),
                                   list(range(len(user_recommendations[:cutoff]))))[0]

    def eval_user_metric(self):
        return {u: self.__pru(u_r, self._cutoff) for u, u_r in self._recommendations.items()}