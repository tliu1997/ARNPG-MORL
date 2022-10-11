import numpy as np
from docplex.mp.model import Model


"""Our code customizes the CMDP code from the paper:
Natural Policy Gradient Primal-Dual Method for Constrained Markov Decision Processes, Dongsheng Ding et al.
"""

"""Natural Policy Gradient Primal-Dual Method with Softmax Parametrization
"""

# Random Seed
np.random.seed(10)
# Problem Setup
gamma = 0.8
s, a = 20, 10

# Randomly generated probability transition matrix P((s,a) -> s') in [0,1]^{|S||A| x |S|}
raw_transition = np.random.uniform(0, 1, size=(s * a, s))
prob_transition = raw_transition / raw_transition.sum(axis=1, keepdims=1)
# Random positive rewards
reward = np.random.uniform(0, 1, size=(s * a))
# Random positive utilities
utility = np.random.uniform(0, 1, size=(s * a))
# Utility constraint offset b
b = 3
# Start state distribution
rho = np.ones(s) / s


def theta_to_policy(theta, s, a):
    """
    :param theta: |S||A| * 1
    :param s: |S|
    :param a: |A|
    :return: |S||A| * 1
    """
    prob = []
    for i in range(s):
        norm = np.sum(np.exp(theta[a * i:a * (i + 1)]))
        for j in range(a * i, a * (i + 1)):
            prob.append(np.exp(theta[j]) / norm)

    return np.asarray(prob)


def get_Pi(prob, s, a):
    """
    :param prob: |S||A| * 1
    :param s: |S|
    :param a: |A|
    :return: |S| * |S||A|
    """
    Pi = np.zeros((s, s * a))
    for i in range(s):
        Pi[i, i * a:(i + 1) * a] = prob[i * a:(i + 1) * a]

    return Pi


def grad_state_action(prob, state, action):
    """
    :param prob: |S||A| * 1
    :param state: 1 * 1
    :param action: 1 * 1
    :return: \nabla_{\theta} \pi_{\theta}(s,a)
    """
    grad = np.zeros(s * a)
    for j in range(0, a):
        if j == action:
            grad[a * state + j] = prob[a * state + j] * (1 - prob[a * state + j])
        else:
            grad[a * state + j] = -prob[a * state + action] * prob[a * state + j]

    return grad


def grad_state(qvals, prob, state):
    grad = np.sum([qvals[state * a + i] * grad_state_action(prob, state, i) for i in range(0, a)], axis=0)
    return grad


def grad(qvals, prob, d_pi):
    grad = np.sum([d_pi[i] * grad_state(qvals, prob, i) for i in range(0, s)], axis=0)
    return grad


def Fisher_info(prob, d_pi):
    """
    :param prob: |S||A| * 1
    :param d_pi: |S| * 1
    :return: Fisher information matrix \nabla_{\theta} \pi_{\theta}(s,a) x {\nabla_{\theta} \pi_{\theta}(s,a)}^T
    """
    qvals_one = np.ones(s * a)
    grad = np.sum([d_pi[i] * grad_state(qvals_one, prob, i) for i in range(0, s)], axis=0)
    fisher = np.outer(grad, grad) + 1e-3 * np.identity(s * a)
    return fisher


def ell(qvals, prob, rho):
    """
    Calculate V from Q value function
    :param qvals: |S||A| * 1
    :param prob: |S||A| * 1
    :param rho: |S| * 1
    :return: V |S| * 1
    """
    V = np.zeros(s)
    for i in range(s):
        V[i] = np.sum([qvals[i * a + j] * prob[i * a + j] for j in range(a)])

    ell = np.dot(V, rho)
    return ell


def proj(scalar):
    offset = 100
    if scalar < 0:
        scalar = 0

    if scalar > offset:
        scalar = offset

    return scalar


# Run policy iteration to get the optimal policy and compute the constraint violation
# Feasibility checking: negative constraint violation leads to the Slater condition
def policy_iter(q_vals, s, a):
    new_policy = np.zeros(s * a)
    for i in range(s):
        idx = np.argmax(q_vals[i * a:(i + 1) * a])
        new_policy[i * a + idx] = 1

    return new_policy


raw_vec = np.random.uniform(0, 1, size=(s, a))
prob_vec = raw_vec / raw_vec.sum(axis=1, keepdims=1)
init_policy = prob_vec.flatten()
curr_policy = np.random.uniform(0, 1, size=(s * a))
new_policy = init_policy

while np.count_nonzero(curr_policy - new_policy) > 0:
    curr_policy = new_policy
    Pi = get_Pi(curr_policy, s, a)
    mat = np.identity(s * a) - gamma * np.matmul(prob_transition, Pi)
    q_vals = np.dot(np.linalg.inv(mat), utility)
    new_policy = policy_iter(q_vals, s, a)

ell_star = ell(q_vals, new_policy, rho)
print('Feasibility checking: constraint violation', b - ell_star)


# calculate the optimal reward via LP
model = Model('CMDP')
# create continuous variables
idx = [(i, j) for i in range(s) for j in range(a)]
x = model.continuous_var_dict(idx)

for i in range(s):
    for j in range(a):
        model.add_constraint(x[i, j] >= 0)

for s_next in range(s):
    model.add_constraint(
        gamma * model.sum(x[i, j] * prob_transition[i * a + j][s_next] for i in range(s) for j in range(a))
        + (1 - gamma) * rho[s_next] == model.sum(x[s_next, a_next] for a_next in range(a)))

model.add_constraint(model.sum(x[i, j] * utility[i * a + j] / (1 - gamma) for i in range(s) for j in range(a)) >= b)

model.maximize(model.sum(x[i, j] * reward[i * a + j] / (1 - gamma) for i in range(s) for j in range(a)))
solution = model.solve()


# NPG with softmax parameterization
N = 300
theta = np.random.uniform(0, 1, size=s * a)
dual = 0
gap = []
violation = []
acc_avg_gap = 0
acc_avg_violation = 0
div_number = 1  # 000
step = 1
dualstep = 1

for k in range(N):
    prob = theta_to_policy(theta, s, a)
    Pi = get_Pi(prob, s, a)
    mat = np.identity(s * a) - gamma * np.matmul(prob_transition, Pi)

    qrvals = np.dot(np.linalg.inv(mat), reward)
    qgvals = np.dot(np.linalg.inv(mat), utility)
    qvals = qrvals + dual * qgvals

    vrvals = ell(qrvals, prob, rho)
    vgvals = ell(qgvals, prob, rho)
    vvals = vrvals + dual * vgvals

    P_theta = np.matmul(Pi, prob_transition)
    d_pi = (1 - gamma) * np.dot(np.transpose((np.linalg.inv(np.identity(s) - gamma * P_theta))), rho)
    MPinverse = np.linalg.pinv(Fisher_info(prob, d_pi))
    gradient = grad(qvals - vvals, prob, d_pi)
    naturalgradient = np.matmul(MPinverse, gradient)

    # primal natural gradient ascent
    # dual projected sub-gradient descent
    theta += step * naturalgradient
    dual = proj(dual - dualstep * (ell(qgvals, prob, rho) - b))

    if k % div_number == 0:
        avg_reward = ell(qrvals, prob, rho)
        avg_violation = b - ell(qgvals, prob, rho)
        # acc_avg_gap = model.objective_value - avg_reward
        # acc_avg_violation = avg_violation
        # print('Average gap:', acc_avg_gap)
        # print('Average violation:', acc_avg_violation)
        # gap.append(acc_avg_gap)
        # violation.append(acc_avg_violation)
        acc_avg_gap += model.objective_value - avg_reward
        acc_avg_violation += avg_violation
        print('Average gap:', acc_avg_gap / (k + 1))
        print('Average volation:', acc_avg_violation / (k + 1))
        gap.append(acc_avg_gap / (k + 1))
        violation.append(acc_avg_violation / (k + 1))


# Saving the data. This can be loaded to make the figure again.
np.savetxt('NPG-PD_gap_s20a10g8b3.txt', gap)
np.savetxt('NPG-PD_violation_s20a10g8b3.txt', violation)