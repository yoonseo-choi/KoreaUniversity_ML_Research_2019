#The CartPole Open Gym AI Exercise

import tensorflow as tf
import numpy as np
import gym

env = gym.make('CartPole-v0')

tf.compat.v1.reset_default_graph()

N = 10
X = 4

obs = tf.compat.v1.placeholder(tf.float32, shape=[None, X], name="input_x")
Wt1 = tf.compat.v1.get_variable(name="Wt1", shape=[X, N], initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(obs, Wt1))
Wt2 = tf.compat.v1.get_variable(name="Wt2", shape=[N, 1], initializer=tf.contrib.layers.xavier_initializer())
value = tf.matmul(layer1, Wt2)
prob = tf.nn.sigmoid(value)

trn_vars = tf.compat.v1.trainable_variables()
env_input = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name="input_y")

#y_est = prob * env_input  #loss function (Mean Square Error)
#y = env_input
#diff = y_est - y
#diff_sq = diff ** 2
#mse_loss = -tf.reduce_mean(diff_sq * adv)

lse = 0.5 * ((env_input - (1 - prob))) ** 2  # Least Square Error
lse_loss = -tf.compat.v1.reduce_mean(lse)

#loglik = tf.compat.v1.log(env_input*(env_input - prob) + (1 - env_input)*(env_input + prob))  #Maximum Likelihood
#loss = -tf.compat.v1.reduce_mean(loglik * adv)

new_grad = tf.compat.v1.gradients(lse_loss, trn_vars)

learn_rate = 0.01
opt = tf.compat.v1.train.AdamOptimizer(learning_rate=learn_rate)
wt1_grad = tf.compat.v1.placeholder(tf.float32, name="batch1_fin")
wt2_grad = tf.compat.v1.placeholder(tf.float32, name="batch2_fin")
btch_grad = [wt1_grad, wt2_grad]
updt_grad = opt.apply_gradients(zip(btch_grad, trn_vars))


gamma = 0.99
def dis_rwd(r):
    dis_r = np.zeros_like(r)
    add = 0
    for t in reversed(range(0, r.size)):
        add = add * gamma + r[t]
        dis_r[t] = add
    return dis_r


reward_sum = 0
batch = 5
ep = 1
tot_ep = 10000
run_rew = None
init = tf.compat.v1.global_variables_initializer()
xs, ys, rs = [], [], []


with tf.compat.v1.Session() as sess1:
    rendering = False
    sess1.run(init)
    observation = env.reset()

    gradBuff = sess1.run(trn_vars)
    for n, grad in enumerate(gradBuff):
        gradBuff[n] = grad * 0  # reset the gradient buffer (the data set for all gradients in given episodes)

    while ep <= tot_ep:
        if reward_sum/batch > 120 or rendering == True:
            env.render()
            rendering = True

        x_in = np.reshape(observation, [1, X])

        pol_prob = sess1.run(prob, feed_dict={obs: x_in})

        if np.random.uniform() < pol_prob:
            action = 1
        else:
            action = 0

        xs.append(x_in)

        if action == 0:
                y_in = 1
        else:
            y_in = 0

        ys.append(y_in)

        observation, reward, done, info = env.step(action)
        reward_sum += reward
        rs.append(reward)

        if done:
            ep += 1
            ep_x = np.vstack(xs)
            ep_y = np.vstack(ys)
            ep_r = np.vstack(rs)
            xs, ys, rs = [], [], []

            dis_epr = dis_rwd(ep_r)
            dis_epr -= np.mean(dis_epr)
            dis_epr /= np.std(dis_epr)  #converting the discounted rewards in the given episodes into a standardized Z-score

            ep_grad = sess1.run(new_grad, feed_dict={obs: ep_x, env_input: ep_y})
            for n, grad in enumerate(ep_grad):
                gradBuff[n] += grad

            if ep % batch == 0:
                sess1.run(updt_grad, feed_dict={wt1_grad: gradBuff[0], wt2_grad: gradBuff[1]})

                for n, grad in enumerate(gradBuff):
                    gradBuff[n] = grad * 0

                if run_rew is None:
                    run_rew = reward_sum
                else:
                    run_rew = run_rew * 0.99 + reward_sum * 0.01

                print("Average Reward is {}. Total Average Reward is {}".format(reward_sum/batch, run_rew/batch))

                if run_rew/batch > 200:
                    print("Task solved in {} episodes".format(np.round(ep, 1)))
                    break

                reward_sum = 0

            observation = env.reset()

print("{} episodes completed".format(ep))
