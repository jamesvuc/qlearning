# flappy_aux.py 

from matplotlib import pyplot as plt
import numpy as np

import sys

sys.path.append('../..')
from qlearning import QLearner

def quantize_x(x):
	"""
	Quantizes the x-distance.
	"""
	if x < 140:
		return int(x) - (int(x) % 10)
	else:
		return int(x) - (int(x) % 70)

def quantize_y(x):
	"""
	Quantizes the y-distance.
	"""
	if x < 180:
		return int(x) - (int(x) % 10)
	else:
		return int(x) - (int(x) % 60)


def learn_rate_func(T, x, a, start=0.10, stop=0.0001, steps=9000.0):
	"""
	Linear annealing of the stepsize, from 0.1 to 0.0001 over 9000 episodes.
	"""
	return max(start-T*(start-stop)/steps, stop)

def greedy_rate_func(T, start=0.01, stop=0.0, steps=9000.0):
	"""
	Linear annealing of the random action probability, 
	from 0.1 to 0.0001 over 9000 episodes.
	"""
	return max(start-T*(start-stop)/steps, stop)

class FlappyLearner(QLearner):
	"""
	Derived class of the QLearner class implementing the
	task-specific methods, etc.
	"""
	def __init__(self,
		n_actions=None,
		discount=1.0,
		init_state=None,
		learn_rate=None,
		p_rand_act=lambda T:0.05,
		memory_size=5,
		mode='off-policy'
	):

		#uses default QMat
		QLearner.__init__(self,
			n_actions=n_actions,
			discount=discount,
			init_state=init_state,
			learn_rate=learn_rate,
			p_rand_act=p_rand_act,
			memory_size=memory_size,
			mode=mode
		)

	#it might be better to provide a TERMINAL identifier...
	def handle_observation(self, o):
		if o[0] != '_':
			return str(o[0])+'_'+str(o[1])+'_'+str(o[2])
		else:
			return None

	def handle_action(self, a):
		return a == 1

	def handle_reward(self, r):
		return r

	# only called during training.
	def callback(self):
		# print the annealing variables
		if self._T % 25 == 0 and self._t ==1 :
			print('learn rate=',self._learn_rate(self._T, None, None))
			print('p(rand)=',self._p_rand_act(self._T))

		# save the current state.
		if self._T % 250 == 0  and self._t == 1:
			print('saving')
			self.save_data('flappy_Q_factors.pkl')

def setup_learner():
	N_actions=2
	actions=np.arange(N_actions)

	learner=FlappyLearner(
		n_actions=N_actions,
		discount=0.99,
		p_rand_act=lambda T:0.0,#comment this for testing
		# p_rand_act=greedy_rate_func, #uncomment this for training only
		learn_rate=learn_rate_func, #doesn't matter if training or not.
		init_state='420_20_0',
		memory_size=20000,
		mode='off-policy'
	)

	# comment for training from scratch
	learner.load_data('flappy_Q_factors.pkl')

	return learner



