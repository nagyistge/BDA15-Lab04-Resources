"""
Posterior probability for machine failure based on counts of defective
widgets.

Based on an example in DeGroot & Schervish, *Probability and Statistics* (2002);
see example 2.3.9, p. 74.

Created 2015-02-05 by Tom Loredo
2015-02-13:  Modified for Lab04
"""

from numpy.testing import assert_approx_equal
from scipy import *


class MachineFailure:
    """
    Infer the probability that a widget production machine is in a failure
    mode, based on counts of defective widgets and knowledge of defect
    and failure rates (probabilities).
    """

    def __init__(self, defect_rate_good, defect_rate_bad, failure_rate):
        """
        Setup a machine failure inference case (with no data).

        Parameters
        ----------
        defect_rate_good : float
            Defect rate for the machine in normal operating mode
        defect_rate_bad : float
            Defect rate for the machine when a key part has failed
        failure_rate : float
            Probability for part failure on startup
        """
        self.drg = defect_rate_good
        self.drb = defect_rate_bad
        self.p_bad = failure_rate
        self.n, self.N = 0, 0

    def update_data(self, N, n):
        """
        Update the defect rate data.

        Parameters
        ----------
        N : int
            Number of widgets produced in a new batch
        n : int
            Number of defective widgets in the batch
        """
        self.N += N
        self.n += n

    def like_g_b(self):
        """
        Return the likelihoods for (good, bad) machine state (bad = failed).
        """
        l_g = self.drg**self.n * (1.-self.drg)**(self.N-self.n)
        l_b = self.drb**self.n * (1.-self.drb)**(self.N-self.n)
        return l_g, l_b

    def p_failed(self):
        """
        Return the probability that the machine has failed.
        """
        l_good, l_bad = self.like_g_b()
        p_good = 1. - self.p_bad
        return self.p_bad*l_bad/ \
            (self.p_bad*l_bad + p_good*l_good)

    def log_like_g_b(self):
        """
        Return the likelihoods for (good, bad) machine state (bad = failed).
        """
        ll_good = self.n*log(self.drg) + (self.N-self.n)*log(1.-self.drg)
        ll_bad = self.n*log(self.drb) + (self.N-self.n)*log(1.-self.drb)
        return ll_good, ll_bad

    def p_failed_ll(self):
        """
        Return the probability that the machine has failed.

        The calculation uses the log likelihood, factoring out the failure
        likelihood.
        """
        ll_good, ll_bad = self.log_like_g_b()
        p_good = 1. - self.p_bad
        return self.p_bad/ \
            (self.p_bad + p_good*exp(ll_good - ll_bad))


# Always good to have tests!

def test_DS_case():
    """
    Duplicate the result from D&S example 2.3.9.
    """
    mf_DS = MachineFailure(.01, .4, .1)
    mf_DS.update_data(6,2)
    p_failed = mf_DS.p_failed()
    assert_approx_equal(1.-p_failed, 0.04, significant=2)

def test_inc_all():
    """
    Check that we get the same result incrementing the data, or
    analyzing it all at once.
    """
    mf_inc = MachineFailure(.01, .4, .1)
    mf_inc.update_data(10, 1)
    mf_inc.update_data(40, 0)
    p_failed_inc = mf_inc.p_failed()
    mf_all = MachineFailure(.01, .4, .1)
    mf_all.update_data(50, 1)
    p_failed_all = mf_all.p_failed()
    assert p_failed_all == p_failed_inc  # should be *exactly* equal

def test_l_ll():
    """
    Check that the likelihood and log likelihood calculations match.
    """
    mf = MachineFailure(.01, .1, .05)
    mf.update_data(50, 1)
    assert_approx_equal(mf.p_failed(), mf.p_failed_ll())


# D&S case:
mf_DS = MachineFailure(.01, .4, .1)
mf_DS.update_data(6, 2)
print 'D&S ex. 2.3.9, failure mode probability:', mf_DS.p_failed()
print

# Case presented in Lec05
mf1 = MachineFailure(.01, .1, .05)
mf1.update_data(10, 1)
print 'Lec05 example, 1 defect in 10:', mf1.p_failed()
mf1.update_data(40, 0)
print 'Update to 1 in 50:', mf1.p_failed()

mf2 = MachineFailure(.01, .1, .05)
mf2.update_data(50, 1)
print 'All at once:', mf2.p_failed()
