import math
import sys
import argparse
from dataclasses import dataclass
from copy import deepcopy
import torch
import time
import math
import json
import numpy as np
import torch
from torch.quasirandom import SobolEngine
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine
import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel, LinearKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
sys.path.append('../')
from mujoco_functions.mujoco_gym_env import MujocoGymEnv
import bo_synthetic_fun


parser = argparse.ArgumentParser(description='BO experiment')
parser.add_argument('-f', '--func', default='ackley', type=str, help='function to optimize')
parser.add_argument('--dim', default=10, type=int, help='dimension of the function')
parser.add_argument('--tr_num', default=1, type=int, help='trust region number')
parser.add_argument('--eval_num', default=5000, type=int, help='evaluation number')
parser.add_argument('--repeat_num', default=20, type=int, help='number of repetition')
parser.add_argument('--gpu_idx', default=0, type=int, help='choose which gpu to use')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

if device == torch.device("cuda"):
    torch.cuda.set_device(args.gpu_idx)



@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 3  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )


def update_state(state, Y_next):
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state

def get_initial_points(dim, n_pts, seed=0):
    X_init = latin_hypercube(n_pts, dim)
    X_init = torch.from_numpy(X_init).to(dtype=dtype, device=device)
    return X_init

def latin_hypercube(n_pts, dim):
    """Basic Latin hypercube implementation with center perturbation."""
    X = np.zeros((n_pts, dim))
    centers = (1.0 + 2.0 * np.arange(0.0, n_pts)) / float(2 * n_pts)
    for i in range(dim):  # Shuffle the center locataions for each dimension.
        X[:, i] = centers[np.random.permutation(n_pts)]

    # Add some perturbations within each box
    pert = np.random.uniform(-1.0, 1.0, (n_pts, dim)) / float(2 * n_pts)
    X += pert
    return X

def generate_batch_multiple_tr(
    state,
    model,  # GP model
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    batch_size,
    n_candidates=None,  # Number of candidates for Thompson sampling
):
    tr_num = len(state)

    for tr_idx in range(tr_num):
        assert X[tr_idx].min() >= 0.0 and X[tr_idx].max() <= 1.0 \
            and torch.all(torch.isfinite(Y[tr_idx]))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))
    dim = X[0].shape[1]
    # Scale the TR to be proportional to the lengthscales
    X_cand = torch.zeros(
        tr_num, n_candidates, dim
        ).to(device=device, dtype=dtype)
    Y_cand = torch.zeros(
        tr_num, n_candidates, batch_size
        ).to(device=device, dtype=dtype)
    tr_lb = torch.zeros(tr_num, dim).to(device=device, dtype=dtype)
    tr_ub = torch.zeros(tr_num, dim).to(device=device, dtype=dtype)
    for tr_idx in range(tr_num):
        x_center = X[tr_idx][Y[tr_idx].argmax(), :].clone()
        try:
            weights = model[tr_idx].covar_module.base_kernel.lengthscale.squeeze().detach()
            weights = weights / weights.mean()
            weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
            tr_lb[tr_idx] = torch.clamp(
                x_center - weights * state[tr_idx].length / 2.0, 0.0, 1.0
                )
            tr_ub[tr_idx] = torch.clamp(
                x_center + weights * state[tr_idx].length / 2.0, 0.0, 1.0
                )
        except: # Linear kernel
            weights = 1
            tr_lb[tr_idx] = torch.clamp(
                x_center - state[tr_idx].length / 2.0, 0.0, 1.0
                )
            tr_ub[tr_idx] = torch.clamp(
                x_center + state[tr_idx].length / 2.0, 0.0, 1.0
                )
        # print(model[0].covar_module.base_kernel.lengthscale)
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
        pert = tr_lb[tr_idx] + (tr_ub[tr_idx] - tr_lb[tr_idx]) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        # prob_perturb = 1
        mask = (
            torch.rand(n_candidates, dim, dtype=dtype, device=device)
            <= prob_perturb
        )
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

        # Create candidate points from the perturbations and the mask        
        X_cand[tr_idx] = x_center.expand(n_candidates, dim).clone()
        X_cand[tr_idx][mask] = pert[mask]

        # Sample on the candidate points
        posterior = model[tr_idx].posterior(X_cand[tr_idx])
        samples = posterior.rsample(sample_shape=torch.Size([batch_size]))
        samples = samples.reshape([batch_size, n_candidates])
        Y_cand[tr_idx] = samples.permute(1,0)
        # recover from normalized value
        Y_cand[tr_idx] = Y[tr_idx].mean() + Y_cand[tr_idx] * Y[tr_idx].std()
        
    # Compare across trust region
    y_cand = Y_cand.detach().cpu().numpy()
    X_next = torch.zeros(batch_size, dim).to(device=device, dtype=dtype)
    tr_idx_next = np.zeros(batch_size)
    for k in range(batch_size):
        i, j = np.unravel_index(np.argmax(y_cand[:, :, k]), (tr_num, n_candidates))
        X_next[k] = X_cand[i, j]
        tr_idx_next[k] = i
        assert np.isfinite(y_cand[i, j, k])  # Just to make sure we never select nan or inf
        # Make sure we never pick this point again
        y_cand[i, j, :] = -np.inf

    return X_next, tr_idx_next


def run(args):
    # set the function
    if args.func in ['ackley', 'rastrigin', 'schaffer', 'levy_montalvo', 'sphere discus', 
                 'schwefel221', 'different_powers', 'bohachevsky']:
        func = getattr(bo_synthetic_fun, args.func)(args.dim, minimize=False)
        func.lb = np.ones(args.dim)*-5
        func.ub = np.ones(args.dim)*10
    elif args.func in [
        'Swimmer', 'Hopper', 'Walker2d', 'HalfCheetah', 'Ant', 'Humanoid', 
        'HumanoidStandup', 'InvertedDoublePendulum', 'InvertedPendulum', 'Reacher'
        ]:
        func = MujocoGymEnv(args.func + '-v2', 10, minimize=False)
        args.dim = func.dim
    else:
        raise ValueError('Unknown function')
    dim = func.dim
    # Same with EA
    args.batch_size = 4 + int(3*np.log(func.dim))
    args.init_num = min(20, 4 + int(3*np.log(func.dim)))
    lb = func.lb
    ub = func.ub
    print('dimension', dim)

    bounds = torch.zeros((2, dim)).to(dtype=dtype, device=device)
    bounds[0, :] = torch.from_numpy(lb).to(dtype=dtype, device=device)
    bounds[1, :] = torch.from_numpy(ub).to(dtype=dtype, device=device)

    max_cholesky_size = float("inf")  # Always use Cholesky
    def eval_objective(x):
        """This is a helper function we use to unnormalize and evalaute a point"""
        x = unnormalize(x, bounds).detach().cpu().numpy()
        fx = func(x)
        return torch.from_numpy(np.asarray(fx)).to(dtype=dtype, device=device)

    all_Y = []
    cur_time = int(time.time())
    for repeat in range(args.repeat_num):
        all_Y.append([])
        # Create turbo state
        state = []
        X_turbo = []
        Y_turbo = []
        for tr_idx in range(args.tr_num):
            X_turbo.append(get_initial_points(dim, args.init_num//args.tr_num))
            Y_turbo.append(torch.tensor(
                [eval_objective(x) for x in X_turbo[tr_idx]], 
                dtype=dtype, device=device
            ).unsqueeze(-1))
            Y_true = deepcopy(Y_turbo[tr_idx])

            Y_list = list(Y_true.detach().cpu().numpy().flatten())
            all_Y[repeat].extend([round(y, 4) for y in Y_list])
            print(f"Init: Total evaluation: {len(all_Y[repeat])},\
                 current Best:{np.max(all_Y[repeat]):.2e}")
            state.append(TurboState(
                    dim, batch_size=args.batch_size, 
                    best_value=max(Y_turbo[tr_idx]).item()
                    ))

        N_CANDIDATES = min(5000, max(2000, 200 * dim)) 

        while len(all_Y[repeat])<args.eval_num:
            # fit GP model
            model = []
            mll = []
            train_Y = []
            for tr_idx in range(args.tr_num):
                train_Y.append((Y_turbo[tr_idx] - Y_turbo[tr_idx].mean()) / Y_turbo[tr_idx].std())
                likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
                if args.func in ['Walker2d','Ant', 'Humanoid', 'HumanoidStandup', 
                'InvertedDoublePendulum', 'InvertedPendulum', 'Reacher']:
                    covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
                                        LinearKernel()
                                    )
                else:
                    covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
                        MaternKernel(nu=2.5, ard_num_dims=dim, 
                        lengthscale_constraint=Interval(0.005, 4.0))
                    )
                model.append(SingleTaskGP(X_turbo[tr_idx], train_Y[tr_idx], 
                covar_module=covar_module, likelihood=likelihood))
                mll.append(ExactMarginalLogLikelihood(model[tr_idx].likelihood, model[tr_idx]))

                # Do the fitting and acquisition function optimization inside the Cholesky context
                with gpytorch.settings.max_cholesky_size(max_cholesky_size):
                    # Fit the model
                    fit_gpytorch_model(mll[tr_idx])
            
            # Get next selection
            X_next, tr_idx_next = generate_batch_multiple_tr(
                    state=state,
                    model=model,
                    X=X_turbo,
                    Y=Y_turbo, # add Y_turbo to recover normalized GP sampling
                    batch_size=args.batch_size,
                    n_candidates=N_CANDIDATES,
                    # use_langevin=use_langevin
                )
            Y_next = torch.tensor(
                [eval_objective(x) for x in X_next], dtype=dtype, device=device
            ).unsqueeze(-1)
            Y_true = deepcopy(Y_next)
            # Update state
            for tr_idx in range(args.tr_num):
                idx_in_tr = np.argwhere(tr_idx_next == tr_idx).reshape(-1)
                if idx_in_tr.shape[0] > 0:
                    state[tr_idx] = update_state(
                        state=state[tr_idx], Y_next=Y_next[idx_in_tr])
                    # Append data
                    X_turbo[tr_idx] = torch.cat((
                        X_turbo[tr_idx], X_next[idx_in_tr]
                        ), dim=0)
                    Y_turbo[tr_idx] = torch.cat((
                        Y_turbo[tr_idx], Y_next[idx_in_tr]
                        ), dim=0)
            Y_list = list(Y_true.detach().cpu().numpy().flatten())
            all_Y[repeat].extend([round(y, 4) for y in Y_list])
            print(f"{func.name} Evaluations: {[x.shape[0] for x in X_turbo]})")
            print(f"Best value: {[round(x.best_value, 4) for x in state]}")
            print(f"TR length: {[x.length for x in state]}")
            print(f"Total evaluation: {len(all_Y[repeat])}, \
                current Best:{np.max(all_Y[repeat]):.2e}")
            # Check restart state
            for tr_idx in range(args.tr_num):
                if state[tr_idx].restart_triggered:
                    X_turbo[tr_idx] = get_initial_points(dim, args.init_num)
                    Y_turbo[tr_idx] = torch.tensor(
                        [eval_objective(x) for x in X_turbo[tr_idx]], 
                        dtype=dtype, device=device
                    ).unsqueeze(-1)

                    Y_true = torch.tensor(
                        [eval_objective(x) for x in X_turbo[tr_idx]], 
                        dtype=dtype, device=device
                    ).unsqueeze(-1)
                    Y_list = list(deepcopy(Y_true).detach().cpu().numpy().flatten())
                    all_Y[repeat].extend([round(y, 4) for y in Y_list])
                    state[tr_idx] = TurboState(dim, batch_size=args.batch_size)
                    print('Trust region ', tr_idx, 'restart')
        # Save data
        with open('../results/' + func.name+ '_' +str(func.dim) +'D_' + \
            str(cur_time) + 'turbo_' + str(args.tr_num) + '.json', 'w') as f:
            json.dump(all_Y, f, ensure_ascii=False)

'''
A running example:

python bo_baseline.py --func ackley --dim 32  --tr_num 1 --eval_num 5000 --repeat_num 30 --gpu_idx 0

'''

if __name__ == '__main__':
    run(args)


