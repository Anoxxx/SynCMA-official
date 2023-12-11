import gym
import numpy as np
import os
import warnings


class MujocoGymEnv():
    # Avaliable environments:
    # https://www.gymlibrary.ml/environments/mujoco/
    def __init__(self, env_name, num_rollouts, minimize=True):
        assert num_rollouts > 0
        self.minimum = minimize
        self.num_rollouts = num_rollouts
        self.mean, self.std = self.get_mean_std(env_name)
        self.env = gym.vector.make(env_name, num_envs=num_rollouts)
        self.__name__ = env_name

        assert self.env.action_space.shape[0] == num_rollouts and self.env.observation_space.shape[0] == num_rollouts
        self.obs_shape = self.env.observation_space.shape[1]
        self.act_shape = self.env.action_space.shape[1]

        self.weight_matrix = self.build_weight_matrix((self.act_shape, self.obs_shape))
        self.dim = self.obs_shape * self.act_shape

        self.lb = -1 * np.ones(self.dim)
        self.ub = 1 * np.ones(self.dim)

    def get_mean_std(self, env_name):
        env_name = env_name.split('-')[0]
        if env_name in ['Ant', 'HalfCheetah', 'Hopper', 'Walker2d', 'Humanoid', 'Swimmer']:
            file_path = os.path.dirname(__file__) + '/trained_policies/' + env_name + '-v1/lin_policy_plus.npz'
            data = np.load(file_path, allow_pickle=True)['arr_0']
            mean = data[1]
            std = data[2]
            return mean, std
        else:
            warnings.warn('No mean and std for this environment')
            return 0, 1

    def reset(self):
        # return self.env.reset(seed=0)
        return self.env.reset(seed=0) # noise

    def step(self, action):
        return self.env.step(action)

    def build_weight_matrix(self, shape):
        return np.random.randn(*shape)

    def get_action(self, obs):
        return np.dot(self.weight_matrix, obs.T)

    def update_weight_matrix(self, updated_weight_matrix):
        if updated_weight_matrix.shape != self.weight_matrix.shape:
            updated_weight_matrix = updated_weight_matrix.reshape(
                self.weight_matrix.shape)
        self.weight_matrix = updated_weight_matrix

    def __call__(self, updated_weight_matrix):
        updated_weight_matrix = np.clip(updated_weight_matrix, self.lb, self.ub)
        assert np.all(updated_weight_matrix <= self.ub) and np.all(
            updated_weight_matrix >= self.lb)
        self.update_weight_matrix(updated_weight_matrix)

        # obs = self.reset()
        obs = self.reset()[0]

        # print(f'obs {obs.shape}')
        done = [False for _ in range(self.num_rollouts)]
        truncated = [False for _ in range(self.num_rollouts)]
        totalReward = [0 for _ in range(self.num_rollouts)]
        # print(f'new one {round(time.time())}')
        while not any(done) and not any(truncated):
            
            obs = (obs - self.mean) / self.std
            action = self.get_action(obs).T
            obs, reward, done,truncated, info = self.step(action)
            # obs, reward, done, info = self.step(action)

            # print(action)
            # print(obs.shape)
            totalReward = [i + j for i, j in zip(totalReward, reward)]
            # print(totalReward)
        if not self.minimum:
            return np.mean(totalReward)
        else:
            return -1 * np.mean(totalReward)

# class Hopper(MujocoGymEnv):
#     def __init__(self, minimize=True):
#         MujocoGymEnv.__init__('Hopper-v2', 10, minimize)

def test_dual_annealing():
    from scipy.optimize import dual_annealing
    mjEnv = MujocoGymEnv("Swimmer-v2", 3)
    ret = dual_annealing(mjEnv, bounds=[(-1, 1) for _ in range(16)])
    print(ret)


def test_differential_evolution():
    from scipy.optimize import differential_evolution
    mjEnv = MujocoGymEnv("Swimmer-v2", 3)
    ret = differential_evolution(mjEnv, bounds=[(-1, 1) for _ in range(16)])
    print(ret)


if __name__ == "__main__":
    # test_dual_annealing()
    # test_differential_evolution()
    func = MujocoGymEnv('Ant-v2', 1, minimize=True)

    x = np.ones(888)
    for i in range(10):
        print(func(x))
