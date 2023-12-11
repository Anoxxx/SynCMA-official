import json
import os
import argparse
import numpy as np
import multiprocessing as mp
from functions.rover_function import *
from syn_cma import SynCMA
from pypop7.benchmarks import base_functions
from mujoco_functions.mujoco_gym_env import MujocoGymEnv
from pypop7.optimizers.es.cmaes import CMAES
from pypop7.optimizers.es.ddcma import DDCMA
from pypop7.optimizers.de import CDE
from pypop7.optimizers.sa import NSA
from pypop7.optimizers.rs import PRS
from baselines.pypop_baselines import bbo_baseline



parser = argparse.ArgumentParser(description='BO experiment')
parser.add_argument('-f', '--func', default='ackley', type=str, help='function to optimize')
parser.add_argument('--dim', default=32, type=int, help='dimension of the function')
parser.add_argument('--optimizer', default='SynCMA', type=str, help='optimizer of the problem')
parser.add_argument('--rep', default=10, type=int, help='repitition number')
parser.add_argument('--eval_num', default=50000, type=int, help='evaluation number')
parser.add_argument('--lam', default=2, type=float, help='lambda_0 of SynCMA')
parser.add_argument('--sigma', default=0.1, type=float, help='sigma of SynCMA')
parser.add_argument('--batch_size', default=None, type=int, help='batch size of SynCMA')
parser.add_argument('--save_x', default=0, type=int, help='Whethers saving x')


# parser.add_argument('-f', '--func', default='RoverPlan', type=str, help='function to optimize')
# parser.add_argument('--dim', default=60, type=int, help='dimension of the function')
# parser.add_argument('--optimizer', default='SynCMA', type=str, help='optimizer of the problem')
# parser.add_argument('--rep', default=50, type=int, help='repitition number')
# parser.add_argument('--eval_num', default=20000, type=int, help='evaluation number')
# parser.add_argument('--lam', default=2, type=float, help='lambda_0 of SynCMA')
args = parser.parse_args()

save_x = True if args.save_x == 1 else False

if args.func in ['ackley', 'rastrigin', 'schaffer', 'levy_montalvo', 'sphere', 'discus', 
                 'schwefel221', 'different_powers', 'bohachevsky','rosenbrock']:
    n = args.dim
    func = getattr(base_functions, args.func)
    problem = {'fitness_function': func,  # define problem arguments
            'ndim_problem': n, 
            'lower_boundary': -5*np.ones((n,)),
            'upper_boundary': 10*np.ones((n,))}
    verbose_freq = 100
elif args.func in [
        'Swimmer', 'Hopper', 'Walker2d', 'HalfCheetah', 'Ant', 'Humanoid', 
        'HumanoidStandup', 'InvertedDoublePendulum', 'InvertedPendulum', 'Reacher'
        ]:
    func = MujocoGymEnv(args.func + '-v2', 10, minimize=True)
    n = func.dim
    problem = {'fitness_function': func,  
                'ndim_problem': n, 
                'lower_boundary': -1*np.ones((n,)),
                'upper_boundary': 1*np.ones((n,)),
                }
    verbose_freq = 1
elif args.func in ['RoverPlan']:
    func = RoverPlan(minimize=True)
    n = args.dim
    problem = {'fitness_function': func,  
                'ndim_problem': n, 
                'lower_boundary': np.zeros((n,)),
                'upper_boundary': np.ones((n,)),
                }
    verbose_freq = 100

args.batch_size = 2 * (problem['ndim_problem']) if args.batch_size is None else args.batch_size


options = {'max_function_evaluations': args.eval_num, 
            'fitness_threshold' : -np.inf,
            'seed_rng': 0,
            'is_restart': False,
            'sigma': args.sigma,
            'n_individuals': args.batch_size,
            'save_interval' : 1,
            'verbose' : verbose_freq
            }

if args.optimizer == 'SynCMA':
    optimizer = SynCMA
    options = {'max_function_evaluations': args.eval_num,  
        'fitness_threshold' : -np.inf,
        'seed_rng': 0,
        'is_restart': False,
            'sigma': args.sigma,
        'save_interval' : 1,
        'verbose' : verbose_freq,
        'n_individuals': args.batch_size,
        'lam_0':args.lam
        #'n_individuals':100
        }
elif args.optimizer == 'CMAES':
    optimizer = bbo_baseline(CMAES, problem, options)
    options = {'max_function_evaluations': args.eval_num,  
        'fitness_threshold' : -np.inf,
        'seed_rng': 0,
        'is_restart': False,
        'sigma': args.sigma,
        'save_interval' : 1,
        'verbose' : verbose_freq,
        'n_individuals':args.batch_size
        #'n_individuals':100
        }
elif args.optimizer == 'DDCMA':
    optimizer = bbo_baseline(DDCMA, problem, options)
    {'max_function_evaluations': args.eval_num,  
        'fitness_threshold' : -np.inf,
        'seed_rng': 0,
        'is_restart': False,
            'sigma': args.sigma,
        'save_interval' : 1,
        'verbose' : verbose_freq,
        # 'n_individuals':100
        'n_individuals':args.batch_size
        }
elif args.optimizer == 'CDE':
    optimizer = bbo_baseline(CDE, problem, options)
    options = {'max_function_evaluations': args.eval_num,  
        'fitness_threshold' : -np.inf,
        'seed_rng': 0,
        'is_restart': False,
            'sigma': args.sigma,
        'save_interval' : 1,
        'verbose' : verbose_freq,
        'n_individuals': args.batch_size
        # 'n_individuals':100
        }
elif args.optimizer == 'NSA':
    optimizer = bbo_baseline(NSA, problem, options)
    options = {'max_function_evaluations': args.eval_num,  
        'fitness_threshold' : -np.inf,
        'seed_rng': 0,
        'is_restart': False,
            'sigma': args.sigma,
        'save_interval' : 1,
        'verbose' : verbose_freq,
        'n_individuals': args.batch_size,
        # 'n_individuals':100,
        'cr': 0.99,
        'temperature': 100}
elif args.optimizer == 'PRS':
    optimizer = bbo_baseline(PRS, problem, options)
    options = {'max_function_evaluations': args.eval_num, 
        'fitness_threshold' : -np.inf,
        'seed_rng': 0,
        'is_restart': False,
            'sigma': args.sigma,
        'save_interval' : 1,
        'verbose' : verbose_freq,
        'n_individuals': args.batch_size
        #'n_individuals':100
        }
else:
    raise NotImplementedError

optimizer.__name__ = args.optimizer


def run_trail(seed) :
    options['seed_rng'] = seed
    # options['seed_rng'] = np.random.randint(1000)

    if args.optimizer == 'SynCMA':
        save_file = f'{optimizer.__name__}_{func.__name__}_{problem["ndim_problem"]}D_{seed}_lam{args.lam}_{options["n_individuals"]}'
    else:
        save_file = f'{optimizer.__name__}_{func.__name__}_{problem["ndim_problem"]}D_{seed}_{options["n_individuals"]}'
        #save_file = f'{optimizer.__name__}_{func.__name__}_{problem["ndim_problem"]}D_{seed}_{100}'
    
    if not os.path.exists('./results'):
        os.mkdir('./results')
    if os.path.exists(f'./results/{save_file}.json'):
        with open(f'./results/{save_file}.json', 'r') as f:
            results = json.load(f)
        print(f'res len {len(results[0])}')
        if len(results[0])>=args.eval_num*0.2:
            print(f'{save_file} has data')
            # if args.optimizer != 'PRS':
            return

    print(f'{save_file} start')
    opt = optimizer(problem, options)
    # try:
    results = opt.optimize()
    print(f'res {results["list_y"][:10]}')
    with open(f'./results/{save_file}.json', 'w') as f:
        json.dump([[round(y, 4) for y in results['list_y']]], f)

    if args.optimizer != 'NSA' and save_x:
        with open(f'./results_x/{save_file}_x.json', 'w') as f:
            x=[]
            for i in range(len(results['list_x'])):
                    current_x = np.round(results['list_x'][i],3)
                    x.append(current_x.tolist())
                    
            json.dump(x, f)

    # except Exception as e:
    #     print(f'{optimizer.__name__} {seed} {e}')

'''
A running example:

python exp.py --optimizer SynCMA --func ackley --dim 32 --eval_num 10000 --rep 100 --lam 2

'''


# if __name__ == '__main__':
#     _rep = np.arange(args.rep)
#     pool = mp.Pool(processes = 20) # set arbitrarily
#     pool.map_async(run_trail, _rep)
#     pool.close()
#     pool.join()


if __name__ == '__main__':
    for j in range(args.rep):
        run_trail(j)