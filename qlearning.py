# qlearning.py
import numpy as np
from abc import ABCMeta, abstractmethod
from collections import deque
import random

import warnings

# def warn_ordering(default=True):
# 	if default:
# 		warnings.simplefilter('always', UserWarning)
warnings.simplefilter('always', UserWarning)

from matplotlib import pyplot as plt

from circbuffer import CircularBuffer
from qfunction import QMat

# the role of QLearner is to wrap the interfaces, and to keep track of transition history
# the optimization/representation of the policy is done with a QFunction() object.
class QLearner(metaclass=ABCMeta):
	"""
	A stateful implementation of the generalized loss-minimization-based 
	offline Q-learning algorithm from https://arxiv.org/abs/1312.5602. 

	The class provides a abstractmethods for the adapters/handlers for an 
	arbitrary environment, as well as the internal machinery to perform Q-learning
	through the act(), observe(), update() cycle.

	The purpose of this class is to keep track of the states of the learner,
	store memory, and interface with the environment. The state-action Q-function
	Q(s,a) is stored, updated, and managed by its own class, QFunction. Hence,
	this implementation is independent of the particular representation of Q(s,a).
	"""
	def __init__(self,
		qfactors=QMat,
		n_actions=2,
		discount=0.9,
		init_state=None,
		p_rand_act=lambda T:0.1,
		learn_rate=None,
		memory_size=0,
		mode='off-policy'
	):
		#set up the MDP
		self._discount=discount
		self._n_actions=n_actions
		self._actions=np.arange(self._n_actions)

		#set up time varianbles
		self._t=0# timesteps per episode
		self._T=0# episodes
		
		# setup the Q-function representation
		# self._Q_factors=QMat(n_actions=self._n_actions, discount=self._discount)
		self._Q_factors=qfactors(n_actions=self._n_actions, discount=self._discount)

		#setup the replay memory. This is implemented using a circular buffer
		# with O(|S|) sampling time for a sample S.
		if memory_size > 0:
			self._memory=CircularBuffer(size=memory_size)
		else:
			self._memory=None

		#set up the epsilon-greedy and learn rate parameters
		self._p_rand_act=p_rand_act
		if learn_rate is None:
			self._learn_rate=lambda T, x,u:0.025
		else:
			self._learn_rate=learn_rate
			
		#setup the states/actions/rewards
		self._prev_state=None
		self._prev_action=None

		if init_state is None:
			raise ValueError('An initial state is required!')
			#the init_state MUST be provided since the first action uses the inital state

		self._curr_state=init_state
		self._curr_action=None

		self._transition_reward=None

		#Eventually, we should add SARSA for on-policy learning.
		if mode not in ['on-policy', 'off-policy']:
			raise ValueError('Acceptable learning modes are "on-policy" and "off-policy".')
		self._learn_mode=mode

		#flags to help ensure that the ordering of the act/observe/update is maintained.
		if self._learn_mode == 'off-policy':
			self._flags={
				'act':False,
				'observe':False,
				'update':True,
				'batch':False
			}
		else:
			raise NotImplementedError("Doesn't support on-policy learning yet.")


	"""
	====REQUIRED USER-SPECIFIED METHODS=====
	"""
	@abstractmethod
	def handle_observation(self):
		"""
		Adapter method mapping an external observed state to an internal state. 
		Must return something that can be processed by the QFunction provided.
		"""
		pass

	# maps an action from one of 'actions' to the external value of the environment
	@abstractmethod
	def handle_action(self):
		"""
		Adapter method mapping an internal acion in [1,...,self._n_actions] to an
		external action that the environment can process.
		"""
		pass

	@abstractmethod
	def handle_reward(self):
		"""
		Adapter method mapping an external reward to one that can be processed by QFunction.
		"""
		pass

	
	"""
	======OPTIONAL USER-SPECIFIED METHODS======
	"""
	def callback(self):
		"""
		Callback called once per either update() or update_batch().
		"""
		pass

	
	"""
	====== PROBLEM-INDEPENDENT METHODS======
	"""
	def act(self):
		"""
		Generate an action according to an epsilon-greedy policy derived from Q.
		Epsilon is called according to the episode self._T.
		"""

		if self._flags['observe']:
			warnings.warn('act() should be called before observe()')

		if (not self._flags['update']) and (not self._flags['batch']):
			warnings.warn('acting without updating!')
		
		# increment time
		self._t+=1

		#update the previous action
		self._prev_action=self._curr_action

		#act randomly with some probability
		if np.random.rand() < self._p_rand_act(self._T):
			# self._curr_action=np.random.choice(self._actions)
			self._curr_action=np.random.randint(self._n_actions)

		else: #act optimally
			self._curr_action=self._Q_factors.get_opt_action(self._curr_state)


		self._flags['update']=False
		self._flags['batch']=False
		self._flags['act']=True

		return self.handle_action(self._curr_action)


	def observe(self, ext_state, ext_reward):
		"""
		Receive, process, and store a state-reward pair. The state transition
		is encoded in the self._prev_state -> self._curr_state variables, hence
		the 'statefulness' of this implementation.
		"""
		if not self._flags['act']:
			warnings.warn('observe() should be called after act!')

		if self._flags['observe']:
			warnings.warn('observe() was called twice without acting/updating!')

		#update the state
		self._prev_state=self._curr_state
		
		#get the new state and map it to the internal representation
		self._curr_state=self.handle_observation(ext_state)

		#get the transition assoicated with _prev_state -> _curr_state
		self._transition_reward=self.handle_reward(ext_reward)

		#add to the memory 
		if self._memory is not None:
			self._memory.put((self._prev_state, self._curr_action, self._transition_reward, self._curr_state))

		self._flags['observe']=True

	def update(self):
		"""
		Get the most-recent transition, and call QFunction.update() to perform the
		learning update.
		"""
		if not (self._flags['act'] and self._flags['observe']):
			if not self._flags['batch']:
				print(self._flags)
				warnings.warn('update() should be called after act() and observe()!')

		if self._learn_mode == 'off-policy':
			# compute the current stepsize.
			alpha=self._learn_rate(self._T, self._prev_state, self._curr_action)
			
			# package the last transition
			last_transition=(self._prev_state, 
							self._curr_action,
							self._transition_reward,
							self._curr_state)
			
			# update the Q(s,a) representation
			self._Q_factors.update([last_transition], alpha)

		self.callback()

		self._flags['act']=False
		self._flags['observe']=False
		self._flags['update']=True


	def update_batch(self, size=10, how='random'):
		"""
		Same as update() but with a batch sample instead.
		"""
		if not (self._flags['act'] and self._flags['observe']):
			if not self._flags['update']:
				print(self._flags)
				warnings.warn('update() should be called after act() and observe()!')

		if how=='random': #Get a random sample from the memory buffer
			transitions=self._memory.sample(size=size)

		elif how == 'all': #get all the transitions, and reset the memory
		# this is used for episodic updates to the parameters.
			transitions=self._memory.sample(size=-1)
			self._memory.reset()
		else:
			raise ValueError('Batch update methods are random or all.')

		alpha=self._learn_rate(self._T, self._prev_state, self._curr_action)
		self._Q_factors.update(transitions, alpha)

		self.callback()

		self._flags['act']=False
		self._flags['observe']=False
		self._flags['batch']=True

	def increment_episode(self):
		self._T+=1
		self._t=0

	def get_opt_policy(self):
		return self._Q_factors.get_opt_policy()
	

	def reset_state(self, state):
		"""
		This should be called at the start of each episode
		"""
		self._curr_state=self.handle_observation(state)

	def save_data(self, path):
		self._Q_factors.save_data(path)

	def load_data(self, path):
		self._Q_factors.load_data(path)

if __name__=='__main__':
	pass