import numpy as np
from pypop7.optimizers.es.es import ES


## Only in function _update_distribution_best does SynCMA act different from the verified version of CMAES in pypop7, except for some initializations.

## We've noticed a recent update with the verified version of CMAES in pypop7
## This makes a SynCMA class that is built upon CMAES to demonstrate seems less 

class SynCMA(ES):
    def __init__(self, problem, options):
        self.options = options
        self.problem = problem
        self.n_log = 1
        ES.__init__(self, problem, options)
        self._mu_eff = None
        self._mu_eff_minus = None
        self._w = None
        self._alpha_cov = 2.0
        self.c_s = None
        self.d_sigma = None
        self.c_c = None
        self.c_1 = None
        self.c_w = None
        self._n_generations = None

    def _set_c_c(self):
        return (4.0 + self._mu_eff/self.ndim_problem)/(self.ndim_problem + 4.0 + 2.0*self._mu_eff/self.ndim_problem)

    def _set_c_w(self):
        # minus 1e-8 for large population size (according to https://github.com/CyberAgentAILab/cmaes)
        return np.minimum(1.0 - self.c_1 - 1e-8, self._alpha_cov*(self._mu_eff + 1.0/self._mu_eff - 2.0) /
                          (np.power(self.ndim_problem + 2.0, 2) + self._alpha_cov*self._mu_eff/2.0))

    def _set_d_sigma(self):
        return 1.0 + self.c_s + 2.0*np.maximum(0.0, np.sqrt((self._mu_eff - 1.0)/(self.ndim_problem + 1.0)) - 1.0)

    def initialize(self, is_restart=False):
        w_apostrophe = np.log((self.n_individuals + 1.0)/2.0) - np.log(np.arange(self.n_individuals) + 1.0)
        self._mu_eff = np.power(np.sum(w_apostrophe[:self.n_parents]), 2)/np.sum(
            np.power(w_apostrophe[:self.n_parents], 2))
        self._mu_eff_minus = np.power(np.sum(w_apostrophe[self.n_parents:]), 2)/np.sum(
            np.power(w_apostrophe[self.n_parents:], 2))
        self.c_s = self.options.get('c_s', (self._mu_eff + 2.0)/(self._mu_eff + self.ndim_problem + 5.0))
        self.d_sigma = self.options.get('d_sigma', self._set_d_sigma())
        self.c_c = self.options.get('c_c', self._set_c_c())
        self.c_1 = self.options.get('c_1', self._alpha_cov/(np.power(self.ndim_problem + 1.3, 2) + self._mu_eff))
        self.c_w = self.options.get('c_w', self._set_c_w()) * 2 # This 'times 2' is to approximate several tricks that used in fine-tuned version of CMAES in pypop 7, when dimension is small, we recomment to omit this 'times 2'.
        w_min = np.min([1.0 + self.c_1/self.c_w, 1.0 + 2.0*self._mu_eff_minus/(self._mu_eff + 2.0),
                        (1.0 - self.c_1 - self.c_w)/(self.ndim_problem*self.c_w)])
        self._w = np.where(w_apostrophe >= 0, 1.0/np.sum(w_apostrophe[w_apostrophe > 0])*w_apostrophe,
                           w_min/(-np.sum(w_apostrophe[w_apostrophe < 0]))*w_apostrophe)
        
        if 'lam_0' in self.options.keys():
            self.lam_0 = self.options['lam_0']
        else:
            self.lam_0 = 2

        self.cc = np.zeros((self.ndim_problem,))
        self.dd = np.zeros((self.ndim_problem,))
        x = np.empty((self.n_individuals, self.ndim_problem))  # offspring
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        p_c = np.zeros((self.ndim_problem,))  #p_c in paper
        p_m = np.zeros((self.ndim_problem,))  #p_m in paper
        self.Q = np.zeros((self.ndim_problem, self.ndim_problem))
        cm = np.eye(self.ndim_problem)  # covariance matrix of Gaussian search distribution
        eig_ve = np.eye(self.ndim_problem)  # eigenvectors of covariance matrix
        eig_va = np.ones((self.ndim_problem,))  # eigenvalues of covariance matrix
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        self._list_initial_mean.append(np.copy(mean))
        self._n_generations = 0
        self.coef = 0

        self.save_list_y = []
        self.save_list_x = []


        return x, mean, p_c, p_m, cm, eig_ve, eig_va, y

    ##############################  Below is the core part, while the above initialization is nearly a copy from CMA-ES implemented in pypop7, defining lot of unused variables

    def iterate(self, x=None, mean=None, eig_ve=None, eig_va=None, y=None, args=None):
        for k in range(self.n_individuals):  # to sampe offspring population
            if self._check_terminations():
                self.save_list_y.extend(y)
                self.save_list_x.extend(np.round(x,3))
                return x, y
            z = self.rng_optimization.standard_normal((self.ndim_problem,))  # Gaussian noise for mutation
            x[k] = mean + self.sigma * np.dot(np.dot(eig_ve, np.diag(eig_va)), z)  # offspring individual, self.sigma is a constant that corresponds the initial learning rate of CMA-ES, that is 0.1 in paper
            y[k] = self._evaluate_fitness(x[k], args)  # fitness
        self.save_list_y.extend(y)
        self.save_list_x.extend(np.round(x,3))
        return x, y

    def _update_distribution_best(self, x=None, mean=None, p_c=None, p_m = None, cm=None, eig_ve=None, eig_va=None, y=None):
        order = np.argsort(y)
        d = (x - mean) / self.sigma # self.sigma is a constant that corresponds the initial learning rate of CMA-ES, that is 0.1 in paper
        wd = np.dot(self._w[:self.n_parents], d[order[:self.n_parents]])
        
        lam = self.lam_0 / (1 + self.lam_0)
        a = np.sqrt(lam)
        b = np.sqrt(1 - lam)

        beta = self.sigma * (wd + p_m * self.lam_0)

        mean += beta # mean update

        c_1 = self.lam_0 * self.c_w # c_1 is lam_0 / z_m
        cm = (1.0 - c_1 - self.c_w) * (cm + np.outer(beta, beta)) + c_1 * np.outer(p_c - beta, p_c - beta)   # implementing (29)
        for i in range(self.n_parents):  # implementing (29)
            cm += self.c_w * self._w[i] * np.outer(d[order[i]] - beta, d[order[i]] - beta) # implementing (29)
        cm += self.c_w * self.Q # implementing (29), Q denotes Q_1 in paper
        cm += self.c_w * self.coef * np.outer(mean, mean) # implementing (29), coef denotes Q_3 in paper
        cm += self.c_w * (np.outer(self.cc, mean) + np.outer(mean, self.cc)) # implementing (29), cc denotes Q_2 in paper

        self.Q = a ** 2 * self.Q # implementing (25)
        for i in range(self.n_parents): # implementing (25)
            self.Q += self.lam_0 * b ** 2 * self._w[i] * np.outer(d[order[i]] - wd, d[order[i]] - wd) # implementing (25)
        pcc = (p_c + mean - beta)
        pdd = (wd + mean - beta)
        self.Q -= self.lam_0 * a * b * (np.outer(pcc, pdd) + np.outer(pdd, pcc)) # implementing (25)
        self.cc = self.cc * a ** 2 - self.lam_0 * (a * (a + b - 2) * pcc + b * (a + b - 2) * pdd) # implementing (26)
        p_c = a * p_c + b * wd + (a + b) * (mean - beta) - mean # implementing (24)
        p_m = a ** 2 * p_m + (1 - a ** 2) * wd - beta # implementing (23)
        self.coef = self.coef * a ** 2 - self.lam_0 * (a - 1) * (b - 1) # implementing (27)
        
        # do eigen decomposition (SVD)
        cm = (cm + np.transpose(cm))/2.0
        eig_va, eig_ve = np.linalg.eigh(cm)
        eig_va = np.sqrt(np.where(eig_va < 0, 1e-8, eig_va))
        cm = np.dot(np.dot(eig_ve, np.diag(np.power(eig_va, 2))), np.transpose(eig_ve))

        return mean, p_c, p_m, cm, eig_ve, eig_va

    def restart_reinitialize(self, x=None, mean=None, p_c=None, p_m = None,
                             cm=None, eig_ve=None, eig_va=None, y=None):
        if ES.restart_reinitialize(self, y):
            x, mean, p_c, p_m, cm, eig_ve, eig_va, y = self.initialize(True)
        return x, mean, p_c, p_m, cm, eig_ve, eig_va, y

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        x, mean, p_c, p_m, cm, eig_ve, eig_va, y = self.initialize()
        while not self._check_terminations():
            # sample and evaluate offspring population
            x, y = self.iterate(x, mean, eig_ve, eig_va, y, args)
            self._print_verbose_info(fitness, y)
            self._n_generations += 1
            mean, p_c, p_m, cm, eig_ve, eig_va = self._update_distribution_best(x, mean, p_c, p_m, cm, eig_ve, eig_va, y)
            if self.is_restart:
                print('suprise!')
                x, mean, p_c, p_m, cm, eig_ve, eig_va, y = self.restart_reinitialize(
                    x, mean, p_c, p_m, cm, eig_ve, eig_va, y)
        results = self._collect(fitness, y, mean)
        results['p_c'] = p_c
        results['p_m'] = p_m
        results['eig_va'] = eig_va
        results['list_y'] = self.save_list_y
        results['list_x'] = self.save_list_x
        # by default, do NOT save covariance matrix of search distribution in order to save memory,
        # owing to its *quadratic* space complexity
        return results
