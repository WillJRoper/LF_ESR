import numpy as np
from esr.fitting.likelihood import Likelihood


class LFLikelihood(Likelihood):

    def __init__(self, data_file, run_name, snap, data_dir="esr_data"):
        """Likelihood class used to fit FLARES LF data
        
        """

        # Intanstiate the parent class.
        super().__init__(self, data_file, data_file, run_name,
                         data_dir=data_dir)

        # Metadata
        self.ylabel = r'$log_10(M / [Mpc / dex / mag])$'    # for plotting

        bins = -np.arange(17.5, 26, 0.25)[::-1]
        bincen = (bins[1:] + bins[:-1]) / 2.
        binwidth = bins[1:] - bins[:-1]

        # Get the data from the master file.
        out, hist_all, err = get_lum_all(tags[ii], bins = bins)
        ok = np.where(hist_all<=5)[0]
        out[ok] = 0.
        hist_all[ok] = 0.
        err[ok] = 0.
        
        phi = out / (3200 ** 3 * binwidth)
        err = err / (3200 ** 3 * binwidth)
        
        self.xvar = bincen
        self.yvar = phi
        self.yerr = err
        self.inv_cov = 1 / self.yerr ** 2

    def get_pred(self, x, a, eq_numpy, **kwargs):
        """
        Return the predicted LF, here we overload the parent method since we
        want the log of the number without requiring this from the model.
        
        Args:
            :x (array): luminosity/magnitude bins.
            :a (list): parameters to subsitute into equation considered
            :eq_numpy (numpy function): function to use which gives H^2
            
        Returns:
            :log_10(N) (float or np.array): the predicted counts in bins
        
        """
        return np.log10(eq_numpy(x, *a))


    def negloglike(self, a, eq_numpy, **kwargs):
        """Negative log-likelihood for a given function
        
        Args:
            :a (list): parameters to subsitute into equation considered
            :eq_numpy (numpy function): function to use which gives H^2
            
        Returns:
            :nll (float): - log(likelihood) for this function and parameters
        
        """
        
        n = self.get_pred(self.xvar, np.atleast_1d(a), eq_numpy)
        if not np.all(np.isreal(n)):
            return np.inf
        nll = np.sum(0.5 * (n - self.yvar) ** 2 * self.inv_cov)
        if np.isnan(nll):
            return np.inf
        return nll

