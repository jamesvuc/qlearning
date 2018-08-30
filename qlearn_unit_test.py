# qlearn_test2.py

import numpy as np
from profilehooks import profile
from matplotlib import pyplot as plt

from qlearning import QLearner

class Environment:
	def __init__(self,
		states=None,
		actions=None,
		init_state=None,
		trans_mat=None,
		reward_func=None):

		self._states=states
		self._actions=actions
		self._trans_mat=trans_mat
		self._gen_reward=reward_func

		self._N_states=self._states.shape[0]
		self._N_actions=self._actions.shape[0]

		self._curr_state=init_state

	def transition(self, a):

		x=self._curr_state

		tmat=self._trans_mat[a]

		x_new=np.random.choice(self._states, p=tmat[x])

		self._curr_state=x_new

		return x_new

	def reward(self, x, a, x_new):
		return self._gen_reward(x, a)

class SimpleLearner(QLearner):
	def __init__(self,
		n_actions=2,
		discount=1.0,
		init_state=None,
		p_rand_act=0.5,
		learn_rate=lambda t,x,u:0.01,
		mode='off-policy'
	):

		QLearner.__init__(self,
			n_actions=n_actions,
			discount=discount,
			init_state=init_state,
			p_rand_act=p_rand_act,
			learn_rate=learn_rate,
			mode=mode
		)

	def handle_observation(self, o):
		return o

	# maps an action from one of 'actions' to the external value of the environment
	def handle_action(self, a):
		return a

	def handle_reward(self, r):
		return r

@profile(sort='time')
def main():

	n_states=2
	n_actions=2

	states=np.arange(n_states)
	actions=np.arange(n_actions)
	
	n_states=3
	n_actions=2

	states=np.arange(n_states)
	actions=np.arange(n_actions)
	
	transmat1=np.array([[0.5, 0.5, 0.0],
						[0.5, 0.5, 0.0],
						[0.7, 0.0, 3.0]])
	transmat1/=np.sum(transmat1, axis=1).T[:,np.newaxis]

	transmat2=np.array([[0.25, 0.5, 0.25],
						[0.5, 0.5, 0.0],
						[0.0, 0.0, 1.0]])
	transmat2/=np.sum(transmat2, axis=1).T[:,np.newaxis]

	init_state=0

	reward_mat=np.array([
		[2,0],
		[0,2],
		[-1,-1]
	])
	reward_func=lambda x,a:reward_mat[x,a]

	print('rewards=')
	for x in states:
		for a in actions:
			print('x=',x,'a=',a,'r=',reward_func(x,a))

	env=Environment(
		states=states,
		actions=actions,
		init_state=init_state,
		trans_mat={0:transmat1, 1:transmat2},
		reward_func=reward_func
	)

	learner=SimpleLearner(
		n_actions=2,
		init_state=init_state,
		p_rand_act=lambda t:1.0,
		discount=0.9,
	)

	cum_reward=np.zeros(200)

	for m in range(0, 200):
		x=init_state
		if m % 20 == 0:
			print('episode m=',m)
		for i in range(0,400):
			a=learner.act()

			x_new=env.transition(a)
			r=env.reward(x, a, x_new)
			cum_reward[m]+=r

			learner.observe(x_new, r)
			learner.update()

			x=x_new

	print(learner.get_opt_policy())

if __name__=='__main__':
	main()