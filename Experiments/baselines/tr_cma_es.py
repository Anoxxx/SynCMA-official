import numpy as np

from pypop7.optimizers.es.es import ES
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

class TRCMAES(ES):
    """Covariance Matrix Adaptation Evolution Strategy (CMAES).

    .. note:: `CMAES` is widely recognized as one of the **State Of The Art (SOTA)** for black-box optimization,
       according to the latest `Nature <https://www.nature.com/articles/nature14544>`_ review of Evolutionary
       Computation.

    Parameters
    ----------
    problem : dict
              problem arguments with the following common settings (`keys`):
                * 'fitness_function' - objective function to be **minimized** (`func`),
                * 'ndim_problem'     - number of dimensionality (`int`),
                * 'upper_boundary'   - upper boundary of search range (`array_like`),
                * 'lower_boundary'   - lower boundary of search range (`array_like`).
    options : dict
              optimizer options with the following common settings (`keys`):
                * 'max_function_evaluations' - maximum of function evaluations (`int`, default: `np.Inf`),
                * 'max_runtime'              - maximal runtime to be allowed (`float`, default: `np.Inf`),
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`);
              and with the following particular settings (`keys`):
                * 'sigma'         - initial global step-size, aka mutation strength (`float`),
                * 'mean'          - initial (starting) point, aka mean of Gaussian search distribution (`array_like`),

                  * if not given, it will draw a random sample from the uniform distribution whose search range is
                    bounded by `problem['lower_boundary']` and `problem['upper_boundary']`.

                * 'n_individuals' - number of offspring, aka offspring population size (`int`, default:
                  `4 + int(3*np.log(problem['ndim_problem']))`),
                * 'n_parents'     - number of parents, aka parental population size (`int`, default:
                  `int(options['n_individuals']/2)`).

    Examples
    --------
    Use the optimizer to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.es.cmaes import CMAES
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5*numpy.ones((2,)),
       ...            'upper_boundary': 5*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'is_restart': False,
       ...            'mean': 3*numpy.ones((2,)),
       ...            'sigma': 0.1}  # the global step-size may need to be tuned for better performance
       >>> cmaes = CMAES(problem, options)  # initialize the optimizer class
       >>> results = cmaes.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"CMAES: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       CMAES: 5000, 9.11305771685916e-09

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/4mysrjwe>`_ for more details.

    Attributes
    ----------
    mean          : `array_like`
                    initial (starting) point, aka mean of Gaussian search distribution.
    n_individuals : `int`
                    number of offspring, aka offspring population size.
    n_parents     : `int`
                    number of parents, aka parental population size.
    sigma         : `float`
                    final global step-size, aka mutation strength.

    References
    ----------
    Hansen, N., 2016.
    The CMA evolution strategy: A tutorial.
    arXiv preprint arXiv:1604.00772.
    https://arxiv.org/abs/1604.00772

    Hansen, N., Müller, S.D. and Koumoutsakos, P., 2003.
    Reducing the time complexity of the derandomized evolution strategy with covariance matrix adaptation (CMA-ES).
    Evolutionary Computation, 11(1), pp.1-18.
    https://direct.mit.edu/evco/article-abstract/11/1/1/1139/Reducing-the-Time-Complexity-of-the-Derandomized

    Hansen, N. and Ostermeier, A., 2001.
    Completely derandomized self-adaptation in evolution strategies.
    Evolutionary Computation, 9(2), pp.159-195.
    https://direct.mit.edu/evco/article-abstract/9/2/159/892/Completely-Derandomized-Self-Adaptation-in

    Hansen, N. and Ostermeier, A., 1996, May.
    Adapting arbitrary normal mutation distributions in evolution strategies: The covariance matrix adaptation.
    In Proceedings of IEEE International Conference on Evolutionary Computation (pp. 312-317). IEEE.
    https://ieeexplore.ieee.org/abstract/document/542381

    See the lightweight implementation of CMA-ES from cyberagent.ai:
    https://github.com/CyberAgentAILab/cmaes
    """
    def __init__(self, problem, options):
        self.options = options
        ES.__init__(self, problem, options)
        assert self.n_individuals >= 2
        self._mu_eff = None
        self.same_count = 0
        self.n_generations, self._n_generations = None, None

    def initialize(self, is_restart=False):
        x = np.empty((self.n_individuals, self.ndim_problem))  # offspring
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        self.pc = np.zeros((self.ndim_problem,1))
        self.ps = np.zeros((self.ndim_problem,1))
        cm = np.eye(self.ndim_problem)  # covariance matrix of Gaussian search distribution
        eig_ve = np.eye(self.ndim_problem)  # eigenvectors of covariance matrix
        eig_va = np.ones((self.ndim_problem,))  # eigenvalues of covariance matrix
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        self._list_initial_mean.append(np.copy(mean))
        self._n_generations = self.n_generations = 0

        # Store old paramaters
        self.old_mean = mean.copy().reshape(self.ndim_problem, 1)
        self.old_sigma = self.sigma
        self.old_cm = cm
        self.old_cov = np.dot(cm, self.sigma)

        self.w = np.log(self.n_parents+0.5) - np.log(np.arange(1, self.n_parents+1))

        self.w = self.w / self.w.sum()

        self._mu_eff = np.sum(self.w) ** 2 / (self.w**2).sum()


        self.save_list_y = []
        self.save_list_x = []

        return x, mean, cm, eig_ve, eig_va, y

    def iterate(self, x=None, mean=None, eig_ve=None, eig_va=None, y=None, args=None):
        for k in range(self.n_individuals):  # to sample offspring population
            if self._check_terminations():
                return x, y
            z = self.rng_optimization.standard_normal((self.ndim_problem,))  # Gaussian noise for mutation
            x[k] = mean + self.sigma*np.dot(np.dot(eig_ve, np.diag(eig_va)), z)  # offspring individual
            y[k] = self._evaluate_fitness(x[k], args)  # fitness
        
        self.save_list_y.extend(y)
        self.save_list_x.extend(np.round(x,3))
        return x, y

    
    def update_distribution(self, x=None, mean=None, cm=None, eig_ve=None, eig_va=None, y=None):

        # Inv
        self.inv_old_cm = np.linalg.inv(self.old_cm)
        self.inv_old_sigma = 1/self.old_sigma
        self.inv_old_cov = self.inv_old_cm * self.inv_old_sigma

        L = np.linalg.cholesky(self.old_cm)
        self.logdet_old_cm = 2*np.sum(np.log(np.diag(L)))

        self.logdet_old_sigma = self.ndim_problem * np.log(self.old_sigma)
        self.logdet_old_cov = self.logdet_old_cm + self.logdet_old_sigma
        Entropy = 0.5 * self.logdet_old_cov + 0.5*self.ndim_problem*(np.log(2*np.pi) + 1)
        expEntropy = np.sqrt(np.exp(Entropy))
        
        # epsilon
        self.epsilon_cm = min(0.2, 1.5*((self._mu_eff+1/self._mu_eff) / ((self.ndim_problem + 2)**2+self._mu_eff)))
        self.epsilon_sigma = (self._mu_eff**2)/(self.ndim_problem*2)
        self.epsilon_mean = 1000

        order = np.argsort(y)
        self.ord_x = x[order]
        self.sample_mean = np.zeros((self.ndim_problem,))
        for i in range(self.n_parents):
            self.sample_mean += self.w[i] * x[order[i]]



        difference = x[order[:self.n_parents]] - self.old_mean.reshape((1, self.ndim_problem))
        self.sample_cov = np.zeros((self.ndim_problem, self.ndim_problem))
        for i in range(self.n_parents):
            d = difference[i].reshape((1, self.ndim_problem))
            
            self.sample_cov += self.w[i] * np.matmul(d.T, d)
        self.new_mean = self.sample_mean.reshape(self.ndim_problem, 1)
        self.cc = (4 + self._mu_eff/self.ndim_problem) / (self.ndim_problem + 4 + 2*self._mu_eff/self.ndim_problem)
        self.cs = (self._mu_eff+2) / (self.ndim_problem+self._mu_eff+5)
        self.c1 = (4*self.ndim_problem) /((self.ndim_problem+1.3)**2+self._mu_eff)

        self.pc = (1-self.cc) *self.pc + 1 * np.sqrt(self.cc*(2-self.cc)*self._mu_eff) * (self.new_mean-self.old_mean) / np.sqrt(self.old_sigma)
        self.ps = ((1-self.cs) * self.ps + np.sqrt(self.cs*(2-self.cs)*self._mu_eff) * (self.new_mean-self.old_mean))


        self.rank1sigma = 1 * (np.matmul(self.ps, self.ps.T))
        self.rank1cm = self.c1 * self.old_sigma * (np.matmul(self.pc, self.pc.T))


        self.new_cm = self.optimiseShapeDualFunction()
        self.new_cm = np.triu(self.new_cm) + np.triu(self.new_cm, 1).T # Ensure symmetry


        self.new_sigma = float(self.optimiseStepSizeDualFunction())
        mean = self.new_mean.flatten()
        cm = self.new_cm
        self.sigma = self.new_sigma


        self.old_mean = mean.reshape((self.ndim_problem, 1))
        self.old_cm = cm
        self.old_sigma = self.new_sigma
        self.old_cov = cm * self.new_sigma

        # old_cm1 = cm
        eig_va, eig_ve = np.linalg.eigh(cm)
        eig_va = np.sqrt(np.where(eig_va < 0, 1e-8, eig_va))
        cm = np.dot(np.dot(eig_ve, np.diag(np.power(eig_va, 2))), np.transpose(eig_ve))
        return mean, cm, eig_ve, eig_va
    

    def compute_KLmean(self, newMean):
        rank1 = np.matmul((newMean - self.old_mean).T, (newMean - self.old_mean))
        KL = 0.5 * (np.abstrace(np.matmul(self.inv_old_cov, rank1)))
        
        return KL

    def computeKLShape(self, newShape):
        k = self.ndim_problem
        L = np.linalg.cholesky(newShape)
        logdet_new_cm = 2*np.sum(np.log(np.diag(L)))
        KL = 0.5 * (np.trace(np.matmul(newShape, np.linalg.inv(self.old_cm))) - k + logdet_new_cm -self.logdet_old_cm)

        return KL

    def optimiseShapeDualFunction(self):
        params = np.array([10])

        rank1 = self.rank1cm
        def dualFunctionShape(lam):
            # print(lam)
            new_cm = (lam * self.old_cm + (self.sample_cov + rank1)/self.old_sigma)/(lam + self.c1 + 1)
            L = np.linalg.cholesky(new_cm)
            logdet_new_cm = 2*np.sum(np.log(np.diag(L)))
            g = -(1+self.c1+lam)*logdet_new_cm - (1/self.old_sigma) \
            * np.trace(
                np.matmul(np.linalg.inv(new_cm),

                              (self.sample_cov+rank1+lam*self.old_cov)

                          ))\
            \
            + lam * (2 * self.epsilon_cm + self.logdet_old_cm + self.ndim_problem)
            return g

        fun = dualFunctionShape
        options = {
            'maxiter': 50000,
            'gtol': 1e-5,
            'xtol': 1e-5
        }
        res = minimize(fun, params, method='trust-constr',
                       # constraints=cons,
                       bounds=[(1e-20, 1e20)],
                       options=options )

        lam = res.x
        new_cm = (lam * self.old_cm + ((self.sample_cov + rank1)/self.old_sigma))/(lam + self.c1 + 1)

        return new_cm
    
    def optimiseStepSizeDualFunction(self):
        params = np.array([1])
        refShape = self.old_cm.copy()
        k = self.ndim_problem
        rank1 = self.rank1sigma

        def dualFunctionStepSize(lam):
            t1 = np.trace(np.matmul(np.linalg.inv(refShape), rank1 + self.sample_cov + lam * self.old_cov))
            new_sigma = ((t1/k) / (1 + lam + 1))
            g = -(1+1+lam)*(k*np.log(new_sigma)) - (t1/new_sigma) + lam * (2 * self.epsilon_sigma + k + k * np.log(self.old_sigma))

            return g

        fun = dualFunctionStepSize
        options = {
            'maxiter': 50000,
            'gtol': 1e-5,
            'xtol': 1e-5
        }
        res = minimize(fun, params, method='trust-constr',
                       # constraints=cons,
                       bounds=[(1e-20, 1e20)],
                       options=options)
        lam = res.x
        t1 = np.trace(np.matmul(np.linalg.inv(refShape), rank1 + self.sample_cov + lam * self.old_sigma * self.old_cm))
        new_sigma = (t1/k)/ (1 + lam + 1)

        return new_sigma

    def restart_reinitialize(self, x=None, mean=None,
                             cm=None, eig_ve=None, eig_va=None, y=None):
        if ES.restart_reinitialize(self, y):
            x, mean, cm, eig_ve, eig_va, y = self.initialize(True)
        return x, mean, cm, eig_ve, eig_va, y

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        x, mean, cm, eig_ve, eig_va, y = self.initialize()
        print
        while True:
            # sample and evaluate offspring population
            self.old_best_y = self.best_so_far_y
            x, y = self.iterate(x, mean, eig_ve, eig_va, y, args)
            if self.best_so_far_y >= self.old_best_y:
                self.same_count += 1
            else:
                self.same_count = 0
            # print(self.same_count)
            if self.same_count >= 20:
                print('early stop')
                break
            if self._check_terminations():
                self.save_list_y.extend(y)
                self.save_list_x.extend(np.round(x,3))
                break
            self._print_verbose_info(fitness, y)
            self.n_generations += 1
            self._n_generations = self.n_generations
            mean, cm, eig_ve, eig_va = self.update_distribution(x, mean, cm, eig_ve, eig_va, y)
            if self.is_restart:
                x, mean, cm, eig_ve, eig_va, y = self.restart_reinitialize(
                    x, mean, cm, eig_ve, eig_va, y)
            
            # print(f'cm {np.linalg.norm(cm)}')
            # if np.linalg.norm(cm) < 1e-10:
            #     break
        results = self._collect(fitness, y, mean)
        results['p_s'] = self.ps
        results['p_c'] = self.pc
        results['eig_va'] = eig_va
        results['list_y'] = self.save_list_y
        results['list_x'] = self.save_list_x
        print('save list y', len(self.save_list_y))
        # by default, do NOT save covariance matrix of search distribution in order to save memory,
        # owing to its *quadratic* space complexity
        return results
