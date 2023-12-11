import numpy as np

def squeeze_and_check(x, size_gt_1=False):
    """Squeeze the input `x` into 1-d `numpy.ndarray`.
        And check whether its number of dimensions == 1. If not, raise a TypeError.
        Optionally, check whether its size > 1. If not, raise a TypeError.
    """
    x = np.squeeze(x)
    if (x.ndim == 0) and (x.size == 1):
        x = np.array([x])
    if x.ndim != 1:
        raise TypeError(f'The number of dimensions should == 1 (not {x.ndim}) after numpy.squeeze(x).')
    if size_gt_1 and not (x.size > 1):
        raise TypeError(f'The size should > 1 (not {x.size}) after numpy.squeeze(x).')
    if x.size == 0:
        raise TypeError(f'the size should != 0.')
    return x

class rastrigin:
    def __init__(self, dim=10, minimize=True):
        self.dim = dim
        self.lb = -5 * np.ones(dim)
        self.ub = 10 * np.ones(dim)
        self.name = 'Rastrigin'
        self.minimize = minimize

    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        f = 10*self.dim + np.sum(x**2-np.cos(2*np.pi*x))
        if not self.minimize:
            return -f
        else:
            return f


class ackley:
    def __init__(self, dim=10, minimize=True):
        self.dim = dim
        self.lb = -5 * np.ones(dim)
        self.ub = 10 * np.ones(dim)
        self.name = 'Ackley'
        self.minimize = minimize

    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        # print(x)
        assert np.all(x <= self.ub) and np.all(x >= self.lb), x
        a, b, c = 20.0, 0.2, 2*np.pi
        f = -a*np.exp(-b*np.sqrt(np.mean(x**2)))
        f -= np.exp(np.mean(np.cos(c*x)))
        f += a + np.exp(1)
        if not self.minimize:
            return -f
        else:
            return f





class schaffer:
    def __init__(self, dim=10, minimize=True):
        self.dim = dim
        self.lb = -5 * np.ones(dim)
        self.ub = 10 * np.ones(dim)
        self.name = 'Schaffer'
        self.minimize = minimize

    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        # print(x)
        assert np.all(x <= self.ub) and np.all(x >= self.lb), x
        x, y = squeeze_and_check(x), 0
        for i in range(x.size - 1):
            xx = np.power(x[i], 2) + np.power(x[i + 1], 2)
            y += np.power(xx, 0.25) * (np.power(np.sin(50 * np.power(xx, 0.1)), 2) + 1)
        f=y
        if not self.minimize:
            return -f
        else:
            return f
        
class levy_montalvo:
    def __init__(self, dim=10, minimize=True):
        self.dim = dim
        self.lb = -5 * np.ones(dim)
        self.ub = 10 * np.ones(dim)
        self.name = 'LevyMontalvo'
        self.minimize = minimize

    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        # print(x)
        assert np.all(x <= self.ub) and np.all(x >= self.lb), x
        x, y = 1 + (1 / 4) * (squeeze_and_check(x) + 1), 0
        for i in range(x.size - 1):
            y += np.power(x[i] - 1, 2) * (1 + 10 * np.power(np.sin(np.pi * x[i + 1]), 2))
        y += 10 * np.power(np.sin(np.pi * x[0]), 2) + np.power(x[-1] - 1, 2)
        f=(np.pi / x.size) * y
        if not self.minimize:
            return -f
        else:
            return f

class bohachevsky:
    def __init__(self, dim=10, minimize=True):
        self.dim = dim
        self.lb = -5 * np.ones(dim)
        self.ub = 10 * np.ones(dim)
        self.name = 'Bohachevsky'
        self.minimize = minimize

    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        # print(x)
        assert np.all(x <= self.ub) and np.all(x >= self.lb), x
        x, y = squeeze_and_check(x), 0
        for i in range(x.size - 1):
            y += np.power(x[i], 2) + 2 * np.power(x[i + 1], 2) -\
                0.3 * np.cos(3 * np.pi * x[i]) - 0.4 * np.cos(4 * np.pi * x[i + 1]) + 0.7
        f = y
        if not self.minimize:
            return -f
        else:
            return f





class diffPowers:
    def __init__(self, dim=10, minimize=True):
        self.dim = dim
        self.lb = -5 * np.ones(dim)
        self.ub = 10 * np.ones(dim)
        self.name = 'DiffPowers'
        self.minimize = minimize

    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        # print(x)
        assert np.all(x <= self.ub) and np.all(x >= self.lb), x
        x = np.abs(squeeze_and_check(x, True))
        y = np.sum(np.power(x, 2 + 4 * np.linspace(0, 1, x.size)))
        f = y
        if not self.minimize:
            return -f
        else:
            return f
        





class schwefel:
    def __init__(self, dim=10, minimize=True):
        self.dim = dim
        self.lb = -5 * np.ones(dim)
        self.ub = 10 * np.ones(dim)
        self.name = 'Schwefel'
        self.minimize = minimize

    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        # print(x)
        assert np.all(x <= self.ub) and np.all(x >= self.lb), x
        y = np.max(np.abs(squeeze_and_check(x)))
        f = y
        if not self.minimize:
            return -f
        else:
            return f
        



class discus:
    def __init__(self, dim=10, minimize=True):
        self.dim = dim
        self.lb = -5 * np.ones(dim)
        self.ub = 10 * np.ones(dim)
        self.name = 'Discus'
        self.minimize = minimize

    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        # print(x)
        assert np.all(x <= self.ub) and np.all(x >= self.lb), x
        x = np.power(squeeze_and_check(x, True), 2)
        y = (10 ** 6) * x[0] + np.sum(x[1:])
        f = y
        if not self.minimize:
            return -f
        else:
            return f




class sphere:
    def __init__(self, dim=10, minimize=True):
        self.dim = dim
        self.lb = -5 * np.ones(dim)
        self.ub = 10 * np.ones(dim)
        self.name = 'Sphere'
        self.minimize = minimize

    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        # print(x)
        assert np.all(x <= self.ub) and np.all(x >= self.lb), x
        
        y = np.sum(np.power(squeeze_and_check(x), 2))
        f = y
        if not self.minimize:
            return -f
        else:
            return f
