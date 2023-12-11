import numpy as np
from pypop7.optimizers.rs import PRS

# Inheritation of original implementation, adding history Y storage
def bbo_baseline(optimizer, problem, options):
    if optimizer != PRS:
        class BBOBaseline(optimizer):
            def __init__(self, problem, options):
                super().__init__(problem, options)
            def initialize(self, *args, **kwargs):
                self.save_list_x = []
                self.save_list_y = []
                return super().initialize(*args, **kwargs)
            def iterate(self, *args, **kwargs):
                output = super().iterate(*args, **kwargs)
                
                if len(output) > 1:
                    x = output[-2]
                    y = output[-1]
                else:
                    x=[]
                    y=output[-1]

                if isinstance(y, np.float64) or isinstance(y, float):
                    self.save_list_x.append(x)
                    self.save_list_y.append(y)
                else:
                    self.save_list_x.extend(x)
                    self.save_list_y.extend(y)
                return output
            
            def optimize(self, *args, **kwargs):
                result = super().optimize(*args, **kwargs)
                result['list_y'] = self.save_list_y
                result['list_x'] = self.save_list_x
                return result
    else:
        class BBOBaseline(PRS):
            def __init__(self, problem, options):
                super().__init__(problem, options)
            def initialize(self, *args, **kwargs):
                self.save_list_x = []
                self.save_list_y = []
                return super().initialize(*args, **kwargs)
            
            def _sample(self, rng):
                x = super()._sample(rng)
                self.save_list_x.append(x)

                return x
            def _evaluate_fitness(self, x, args=None):
                y = super()._evaluate_fitness(x, args)
                self.save_list_y.append(y)
                return y
                


                return super()._collect(fitness, y)
            def optimize(self, fitness_function=None, args=None):
                result = super().optimize(fitness_function, args)
                result['list_y'] = self.save_list_y
                result['list_x'] = self.save_list_x
                return result

    return BBOBaseline
