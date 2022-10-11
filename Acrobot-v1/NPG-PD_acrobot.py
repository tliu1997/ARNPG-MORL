import torch.nn as nn
import torch.nn.functional as F
import collections
import copy
import torch
import gym
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
import numpy as np
from utils.torch_utils import use_cuda, Tensor, Variable, ValueFunctionWrapper
import utils.math_utils as math_utils
import matplotlib.pyplot as plt

"""Our code customizes the CMDP code from the paper:
CRPO: A New Approach for Safe Reinforcement Learning with Convergence Guarantee, Tengyu Xu et al.
"""

"""NPG-PD Method with Softmax Parametrization
"""


class DQNSoftmax(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNSoftmax, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.head = nn.Linear(128, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = F.relu((self.fc1(x)))
        out = F.relu(self.fc2(out.view(out.size(0), -1)))
        out = self.softmax(self.head(out))
        return out


class DQNRegressor(nn.Module):
    def __init__(self, input_size):
        super(DQNRegressor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        out = F.relu((self.fc1(x)))
        out = F.relu(self.fc2(out.view(out.size(0), -1)))
        out = self.head(out)
        return out


def sample_action_from_policy(observation, policy_model):
    observation_tensor = Tensor(observation).unsqueeze(0)
    probabilities = policy_model(Variable(observation_tensor, requires_grad=True))
    action = probabilities.multinomial(1)
    return action, probabilities


def flatten(l):
    return [item for sublist in l for item in sublist]


def reward_function(cos1, sin1, cos2, sin2):
    if sin1 * sin2 - cos1 * cos2 - cos1 > -0.5:
        return 1.0
    else:
        return 0.0


def constrain_I(theta1_dot, action, penality):
    if theta1_dot < penality and action != 0:
        return -1
    else:
        return 0


def constrain_II(theta2_dot, action, penality2):
    if theta2_dot < penality2 and action != 0:
        return -1
    else:
        return 0


def sample_trajectories(env, gamma, episodes, length, policy_model, penality, penality2):
    paths = []
    episodes_so_far = 0
    entropy = 0

    while episodes_so_far < episodes:
        episodes_so_far += 1
        observations, actions, rewards, costs, costs2, action_distributions = [], [], [], [], [], []
        observation = env.reset()
        length_so_far = 0
        done = False
        while length_so_far < length:
            if done: observation = env.reset()
            observations.append(observation)
            action, action_dist = sample_action_from_policy(observation, policy_model)
            actions.append(action)
            action_distributions.append(action_dist)
            entropy += -(action_dist * action_dist.log()).sum()

            reward = reward_function(observation[0], observation[1], observation[2], observation[3])
            cost = constrain_I(observation[4], action, penality)
            cost2 = constrain_II(observation[5], action, penality2)

            rewards.append(reward)
            costs.append(cost)
            costs2.append(cost2)

            # next step
            observation, _, done, _ = env.step(action[0, 0].item())  ## change I
            length_so_far += 1
        # print("episode: ", episodes_so_far, "length: ", length)

        path = {"observations": observations,
                "actions": actions,
                "rewards": rewards,
                "costs": costs,
                "costs2": costs2,
                "action_distributions": action_distributions}
        paths.append(path)

    observations = flatten([path["observations"] for path in paths])
    discounted_rewards = flatten([math_utils.discount(path["rewards"], gamma) for path in paths])
    total_reward = sum(flatten([path["rewards"] for path in paths])) / episodes
    ## add for cost
    discounted_costs = flatten([math_utils.discount(path["costs"], gamma) for path in paths])
    total_cost = sum(flatten([path["costs"] for path in paths])) / episodes
    discounted_costs2 = flatten([math_utils.discount(path["costs2"], gamma) for path in paths])
    total_cost2 = sum(flatten([path["costs2"] for path in paths])) / episodes

    actions = flatten([path["actions"] for path in paths])
    action_dists = flatten([path["action_distributions"] for path in paths])
    entropy = entropy / len(actions)

    return observations, np.asarray(discounted_rewards), total_reward, np.asarray(discounted_costs), total_cost, \
           np.asarray(discounted_costs2), total_cost2, actions, action_dists, entropy


def mean_kl_divergence(model, policy_model, observations):
    observations_tensor = torch.cat(
        [Variable(Tensor(observation)).unsqueeze(0) for observation in observations])
    actprob = model(observations_tensor).detach() + 1e-8
    old_actprob = policy_model(observations_tensor)
    return torch.sum(old_actprob * torch.log(old_actprob / actprob), 1).mean()


def hessian_vector_product(vector, policy_model, observations, cg_damping):
    policy_model.zero_grad()
    mean_kl_div = mean_kl_divergence(policy_model, policy_model, observations)
    kl_grad = torch.autograd.grad(
        mean_kl_div, policy_model.parameters(), create_graph=True)
    kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
    grad_vector_product = torch.sum(kl_grad_vector * vector)
    grad_grad = torch.autograd.grad(
        grad_vector_product, policy_model.parameters())
    fisher_vector_product = torch.cat(
        [grad.contiguous().view(-1) for grad in grad_grad]).data
    return fisher_vector_product + (cg_damping * vector.data)


def conjugate_gradient(policy_model, observations, cg_damping, b, cg_iters, residual_tol):
    p = b.clone().data
    r = b.clone().data
    x = np.zeros_like(b.data.cpu().numpy())
    rdotr = r.double().dot(r.double())
    for _ in range(cg_iters):
        z = hessian_vector_product(Variable(p), policy_model, observations, cg_damping).squeeze(0)
        v = rdotr / p.double().dot(z.double())
        # x += v * p.cpu().numpy()
        x += v.cpu().numpy() * p.cpu().numpy()  # change II
        r -= v * z
        newrdotr = r.double().dot(r.double())
        mu = newrdotr / rdotr
        p = r + mu * p
        rdotr = newrdotr
        if rdotr < residual_tol:
            break
    return x


def surrogate_loss(theta, policy_model, observations, actions, advantage):
    new_model = copy.deepcopy(policy_model)
    vector_to_parameters(theta, new_model.parameters())
    observations_tensor = torch.cat(
        [Variable(Tensor(observation)).unsqueeze(0) for observation in observations])
    prob_new = new_model(observations_tensor).gather(
        1, torch.cat(actions)).data
    prob_old = policy_model(observations_tensor).gather(
        1, torch.cat(actions)).data + 1e-8
    return -torch.mean((prob_new / prob_old) * advantage)


def linesearch(x, policy_model, observations, actions, advantage, fullstep, expected_improve_rate):
    accept_ratio = .1
    max_backtracks = 10
    fval = surrogate_loss(x, policy_model, observations, actions, advantage)
    for (_n_backtracks, stepfrac) in enumerate(.5 ** np.arange(max_backtracks)):
        print("Search number {}...".format(_n_backtracks + 1))
        xnew = x.data.cpu().numpy() + stepfrac * fullstep
        newfval = surrogate_loss(Variable(torch.from_numpy(xnew)), policy_model, observations, actions, advantage)
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        if ratio > accept_ratio and actual_improve > 0:
            return Variable(torch.from_numpy(xnew)), stepfrac
    return x, stepfrac


def normalize(advantage, cost_advantage, cost_advantage2, lagrange1, lagrange2):
    normal_advantage = advantage + lagrange1 * cost_advantage + lagrange2 * cost_advantage2
    normal_advantage = (normal_advantage - normal_advantage.mean()) / (normal_advantage.std() + 1e-8)
    return normal_advantage


def step(env, policy_model, value_function_model, cost_value_function_model, cost_value_function_model2, gamma,
         episodes, length, batch_size, max_kl, cg_iters, residual_tol, cg_damping, ent_coeff, penality, penality2,
         limit, limit2, lagrange1, lagrange2):
    # Generate rollout
    all_observations, all_discounted_rewards, total_reward, all_discounted_costs, total_cost, \
    all_discounted_costs2, total_cost2, all_actions, all_action_dists, \
    entropy = sample_trajectories(env, gamma, episodes, length, policy_model, penality, penality2)

    num_batches = len(all_actions) // batch_size + 1
    for batch_num in range(num_batches):
        print("Processing batch number {}".format(batch_num + 1))
        observations = all_observations[batch_num * batch_size:(batch_num + 1) * batch_size]
        discounted_rewards = all_discounted_rewards[batch_num * batch_size:(batch_num + 1) * batch_size]
        discounted_costs = all_discounted_costs[batch_num * batch_size:(batch_num + 1) * batch_size]
        discounted_costs2 = all_discounted_costs2[batch_num * batch_size:(batch_num + 1) * batch_size]
        actions = all_actions[batch_num * batch_size:(batch_num + 1) * batch_size]
        action_dists = all_action_dists[batch_num * batch_size:(batch_num + 1) * batch_size]

        # Calculate the advantage of each step by taking the actual discounted rewards seen
        # and subtracting the estimated value of each state
        baseline = value_function_model.predict(observations).data
        cost_baseline = cost_value_function_model.predict(observations).data
        cost_baseline2 = cost_value_function_model2.predict(observations).data
        discounted_rewards_tensor = Tensor(discounted_rewards).unsqueeze(1)
        discounted_costs_tensor = Tensor(discounted_costs).unsqueeze(1)
        discounted_costs_tensor2 = Tensor(discounted_costs2).unsqueeze(1)
        advantage = discounted_rewards_tensor - baseline
        cost_advantage = discounted_costs_tensor - cost_baseline
        cost_advantage2 = discounted_costs_tensor2 - cost_baseline2

        # Normalize the advantage
        lagrange_advantage = normalize(advantage, cost_advantage, cost_advantage2, lagrange1, lagrange2)

        # Calculate the surrogate loss as the elementwise product of the advantage and the probability ratio of actions taken
        new_p = torch.cat(action_dists).gather(1, torch.cat(actions))
        old_p = new_p.detach() + 1e-8
        prob_ratio = new_p / old_p
        # surrogate_loss = - torch.mean(prob_ratio * Variable(advantage)) - (ent_coeff * entropy)
        # cost_surrogate_loss = - torch.mean(prob_ratio * Variable(cost_advantage)) - (ent_coeff * entropy)
        # cost_surrogate_loss2 = - torch.mean(prob_ratio * Variable(cost_advantage2)) - (ent_coeff * entropy)

        lagrange_surrogate_loss = - torch.mean(prob_ratio * Variable(lagrange_advantage)) - (ent_coeff * entropy)

        # Calculate the gradient of the surrogate loss
        policy_model.zero_grad()
        lagrange_surrogate_loss.backward(retain_graph=True)

        policy_gradient = parameters_to_vector([v.grad for v in policy_model.parameters()]).squeeze(0)

        if policy_gradient.nonzero().size()[0]:
            # Use conjugate gradient algorithm to determine the step direction in theta space
            step_direction = conjugate_gradient(policy_model, observations, cg_damping, -policy_gradient, cg_iters,
                                                residual_tol)
            step_direction_variable = Variable(torch.from_numpy(step_direction))

            # Do line search to determine the stepsize of theta in the direction of step_direction
            shs = .5 * step_direction.dot(
                hessian_vector_product(step_direction_variable, policy_model, observations, cg_damping).cpu().numpy().T)
            lm = np.sqrt(shs / max_kl)
            fullstep = step_direction / lm
            # gdotstepdir = -policy_gradient.dot(step_direction_variable).data[0]
            gdotstepdir = -policy_gradient.dot(step_direction_variable).data.item()  # change III
            # gdotstepdir = gdotstepdir_mid.item()

            theta, stepfrac = linesearch(parameters_to_vector(policy_model.parameters()), policy_model, observations,
                                         actions,
                                         lagrange_advantage, fullstep, gdotstepdir / lm)

            ## Update dual variable
            # lagrange1 = max(lagrange1 + (stepfrac / lm) * (limit - total_cost), 0.0)
            # lagrange2 = max(lagrange2 + (stepfrac / lm) * (limit2 - total_cost2), 0.0)

            lagrange1 = max(lagrange1 + 0.0005 * (limit - total_cost),
                            0.0)  # previous 0.0002 looks good, 0.0005 still ok
            lagrange2 = max(lagrange2 + 0.0005 * (limit2 - total_cost2), 0.0)

            # Fit the estimated value function to the actual observed discounted rewards
            ev_before = math_utils.explained_variance_1d(baseline.squeeze(1).cpu().numpy(), discounted_rewards)
            cost_ev_before = math_utils.explained_variance_1d(cost_baseline.squeeze(1).cpu().numpy(), discounted_costs)
            cost_ev_before2 = math_utils.explained_variance_1d(cost_baseline2.squeeze(1).cpu().numpy(),
                                                               discounted_costs2)
            value_function_model.zero_grad()
            value_fn_params = parameters_to_vector(value_function_model.parameters())
            cost_value_function_model.zero_grad()
            cost_value_fn_params = parameters_to_vector(cost_value_function_model.parameters())
            cost_value_function_model2.zero_grad()
            cost_value_fn_params2 = parameters_to_vector(cost_value_function_model2.parameters())

            value_function_model.fit(observations, Variable(discounted_rewards_tensor))
            cost_value_function_model.fit(observations, Variable(discounted_costs_tensor))
            cost_value_function_model2.fit(observations, Variable(discounted_costs_tensor2))

            ev_after = math_utils.explained_variance_1d(
                value_function_model.predict(observations).data.squeeze(1).cpu().numpy(), discounted_rewards)
            cost_ev_after = math_utils.explained_variance_1d(
                cost_value_function_model.predict(observations).data.squeeze(1).cpu().numpy(), discounted_costs)
            cost_ev_after2 = math_utils.explained_variance_1d(
                cost_value_function_model2.predict(observations).data.squeeze(1).cpu().numpy(), discounted_costs2)

            if ev_after < ev_before or np.abs(ev_after) < 1e-4:
                vector_to_parameters(value_fn_params, value_function_model.parameters())

            if cost_ev_after < cost_ev_before or np.abs(cost_ev_after) < 1e-4:
                vector_to_parameters(cost_value_fn_params, cost_value_function_model.parameters())

            if cost_ev_after2 < cost_ev_before2 or np.abs(cost_ev_after2) < 1e-4:
                vector_to_parameters(cost_value_fn_params2, cost_value_function_model2.parameters())

            # Update parameters of policy model
            old_model = copy.deepcopy(policy_model)
            old_model.load_state_dict(policy_model.state_dict())
            if any(np.isnan(theta.data.cpu().numpy())):
                print("NaN detected. Skipping update...")
            else:
                vector_to_parameters(theta, policy_model.parameters())

            kl_old_new = mean_kl_divergence(old_model, policy_model, observations)
            diagnostics = collections.OrderedDict(
                [('Total Reward', total_reward), ('Total Cost', -1 * total_cost), ('Total Cost2', -1 * total_cost2),
                 ('KL Old New', kl_old_new.data.item()), ('Entropy', entropy.data.item()), ('EV Before', ev_before),
                 ('EV After', ev_after)])
            for key, value in diagnostics.items():
                print("{}: {}".format(key, value))

        else:
            print("Policy gradient is 0. Skipping update...")

    return total_reward, total_cost, total_cost2, lagrange1, lagrange2


def main():
    env = gym.make("Acrobot-v1")

    ## Hyperparameter
    value_function_lr = 1
    gamma = 0.98
    episodes = 10
    length = 500
    max_kl = 0.003  ## suggest range: 0.001 to 0.1 for fine tune
    cg_damping = 0.001
    cg_iters = 10
    residual_tol = 1e-10
    ent_coeff = 0.00
    batch_size = 5100

    STEP = 400

    ## Constrains
    penality = 0.0
    penality2 = 0.0

    limit = -50  # I: 50
    limit2 = -50  # I: 50

    ## Initial dual vairable
    lagrange1, lagrange2 = 0.0, 0.0

    ## Initial neural network
    policy_model = DQNSoftmax(6, 3)
    value_function_model = DQNRegressor(6)
    value_function_model = ValueFunctionWrapper(value_function_model, value_function_lr)
    cost_value_function_model = DQNRegressor(6)
    cost_value_function_model = ValueFunctionWrapper(cost_value_function_model, value_function_lr)
    cost_value_function_model2 = DQNRegressor(6)
    cost_value_function_model2 = ValueFunctionWrapper(cost_value_function_model2, value_function_lr)

    ## Cuda identity
    if use_cuda:
        policy_model.cuda()
        value_function_model.cuda()
        cost_value_function_model.cuda()
        cost_value_function_model2.cuda()

    ## Training neural network
    iterations = []
    results = []
    violations = []
    violations2 = []
    results_avg = []
    violations_avg = []
    violations2_avg = []
    result_avg = 0
    violation_avg = 0
    violation2_avg = 0
    for iteration in range(STEP):
        result, violation, violation2, lagrange1, lagrange2 = step(env, policy_model, value_function_model,
                                                                   cost_value_function_model,
                                                                   cost_value_function_model2, gamma,
                                                                   episodes, length, batch_size, max_kl, cg_iters,
                                                                   residual_tol, cg_damping, ent_coeff,
                                                                   penality, penality2, limit, limit2, lagrange1,
                                                                   lagrange2)

        print('lagrange1, lagrange2 is', lagrange1, lagrange2)

        results.append(result)
        violations.append(-1 * violation)
        violations2.append(-1 * violation2)

        result_avg = (iteration / (iteration + 1)) * result_avg + (1 / (iteration + 1)) * result
        violation_avg = (iteration / (iteration + 1)) * violation_avg + (1 / (iteration + 1)) * (-1 * violation)
        violation2_avg = (iteration / (iteration + 1)) * violation2_avg + (1 / (iteration + 1)) * (-1 * violation2)

        results_avg.append(result_avg)
        violations_avg.append(violation_avg)
        violations2_avg.append(violation2_avg)

        iterations.append(iteration * episodes)
        if (iteration + 1) % 50 == 0:
            print("step:", iteration + 1, "test result:", result, "test violation:", -1 * violation, "test violation2:",
                  -1 * violation2)

    constraint1 = [-1 * limit for i in range(STEP)]
    constraint2 = [-1 * limit2 for i in range(STEP)]

    plt.figure()
    plt.plot(iterations, results, color='r', linestyle='-', label='L-TRPO reward')
    plt.plot(iterations, violations, color='b', linestyle='-', label='L-TRPO cost')
    plt.plot(iterations, violations2, color='g', linestyle='-', label='L-TRPO cost2')

    # plt.plot(iterations, results_avg, color='r', linestyle='-.', label='C-TRPO average reward')
    plt.plot(iterations, violations_avg, color='b', linestyle='-.', label='L-TRPO average cost')
    plt.plot(iterations, violations2_avg, color='g', linestyle='-.', label='L-TRPO average cost2')

    plt.plot(iterations, constraint1, color='b', linestyle='--')
    plt.plot(iterations, constraint2, color='g', linestyle='--')
    plt.legend(loc='upper left')
    plt.xlabel('# of Episodes')
    plt.ylabel('Reward/Cost')
    plt.savefig('NPG-PD.png')
    plt.show()

    np.savetxt('NPG-PD_reward.txt', results)
    np.savetxt('NPG-PD_cost.txt', violations)
    np.savetxt('NPG-PD_cost2.txt', violations2)


if __name__ == '__main__':
    main()
