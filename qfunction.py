from abc import ABCMeta, abstractmethod
import numpy as np

from copy import copy

import pickle, os, warnings

#implemented as a stateful class in case we need to store information from one
# state to the next. This is because we are doing online optimization.

"""
	Some online optimizers. These are just stateful versions of the 
	autograd.optimizers found at 
	https://github.com/HIPS/autograd/blob/master/autograd/misc/optimizers.py
"""

class SGD:
	def __init__(self):
		pass

	def step(self, x, g, alpha):
		return x-alpha*g


class SGDMomentum:
	def __init__(self, mass=0.9):
		self._mass=mass
		self._v=None

	def step(self, x, g, alpha):
		if self._v == None:
			self._v=np.zeros(*x.shape)

		self._v= self._mass * self._v - (1.0 - self._mass) * g

		return x + alpha * self._v

class Adam:
	def __init__(self,
		b1=0.9,
		b2=0.999,
		eps=10**-8,
	):
	
		self._b1, self._b2 = b1, b2
		self._eps=eps
		self._m = None
		self._v = None
		self._mhat=None
		self._vhat=None

		self._t=0

	def step(x, g, alpha):
		if self._m is None:
			self._m=np.zeros(*x.shape)
			self._v=np.zeros(*x.shape)

		self._m = (1 - self._b1) * g + self._b1 * self._m
		self._v = (1 - self._b2) * (g**2) + self._b2 * self._v

		self._mhat= self._m / (1-self._b1 **(self._t + 1 ))
		self._vhat= self._v / (1-self._b2 **(self._t + 1 ))

		return x - alpha * self._mhat/(np.sqrt(self._vhat) + eps)

class QFunction(metaclass=ABCMeta):
	"""
		A base class for the representation of the Q-funciton Q(s,a).
		This provides a stateless (states are handled by QLeanrer()) 'skeleton' 
		class for a general parameterized representation of Q(s,a; theta) suitable 
		for gradient-based loss minimization as described in 
		https://arxiv.org/abs/1312.5602.

		This generalizes the tabular form of Q(s,a) as shown in the example below.
	"""
	def __init__(self, optimizer=SGD(), n_actions=2):
		self._optimizer=optimizer
		self._n_actions=n_actions

	"""
	====REQUIRED USER-SPECIFIED METHODS=====
	"""
	@abstractmethod
	def get_params(self, transitions):
		"""
		Access method for getting the parameters (i.e. theta) of the Q-function.
		The parameters may be sparse (e.g. in a hashtable representation), in which
		case only those parameters corresponding to the transitions provided should
		be fetched.
		"""
		pass

	@abstractmethod
	def set_params(self, params, transitions):
		"""
		Access method for setting the parameters (i.e. theta) of the Q-function.
		Similar sparseness consideratins apply as above.
		"""
		pass


	@abstractmethod
	def loss_grad(self, transitions):
		"""
		Stochastic estimate of the loss gradient, with the expectation 
		taken via MCMC from the transitions
		"""
		pass
	
	@abstractmethod
	def get_opt_action(self, state):
		"""
		Return the action of the greedy policy derived from Q(s,a; theta) in state 'state'.
		"""
		pass

	"""
	======OPTIONAL USER-SPECIFIED METHODS======
	"""
	def get_opt_policy(self):
		"""
		Returns a dictionary {state:greedy action for state in states}.
		"""
		pass

	def save_data(self, path):
		"""
		Store the current parameter values at 'path'
		"""
		pass

	def load_data(self, path):
		"""
		Load and set the current parameter values at 'path'.

		"""
		pass

	"""
	====== PROBLEM-INDEPENDENT METHODS======
	"""
	def update(self, transitions, alpha):
		"""
		Do a gradient update of the parameters (possibly dependent on trasnitions)
		with learning rate alpha
		"""
		#evaluate the gradient of the loss
		gradient=self.loss_grad(transitions)

		#get the old parameter values (can be sparse i.e. depending on transitions)
		old_params=self.get_params(transitions)

		new_params=[]
		if type(gradient) is type([1,2,3]): #gradient update cannot be vectorized.
			for i,g in enumerate(gradient):
				# new_params=[self._optimizer.step(old_param, g, alpha) for old_param in old_params]
				new_params.append(self._optimizer.step(old_params[i], g, alpha))
			
		else: #gradient update can be vectorized
			new_params=self._optimizer.step(old_params, gradient, alpha)
		
		self.set_params(transitions, new_params)

"""
=================================================
An example class showing how to form a tabular Q function from the base class above.
"""

class QMat(QFunction):
	def __init__(self, n_actions=2, discount=0.99):
		self._n_actions=n_actions
		self._actions=np.arange(self._n_actions)
		
		self._beta=discount

		self._Q={}

		#use default optimizer
		QFunction.__init__(self, n_actions=n_actions)

	#helper method
	def _get_check_state(self, state):
		try:
			return self._Q[state]
		except KeyError:
			self._Q[state]=np.zeros(self._n_actions)

		return self._Q[state]

	"""
	=====Mandatory methods=====
	"""
	#assume that transition=(x,a,r,x_new)
	def get_params(self, transitions):
		params=[]
		for state,action in [(trans[0],trans[1]) for trans in transitions]:
			Q_x=self._get_check_state(state)
			params.append(Q_x[action])

		return params

	def set_params(self, transitions, params):
		if not len(params) == len(transitions):
			raise ValueError('params and transition lengths differ!')

		for i,param in enumerate(params):
			state=transitions[i][0]
			action=transitions[i][1]

			#this doesn't conform to the safe state-access paradigm,
			# but I want it to throw an error if the state doesn't exist
			self._Q[state][action]=copy(param)

	def get_opt_action(self, state):
		Q_x=self._get_check_state(state)
		return self._actions[np.argmax(Q_x)]

	def loss_grad(self, transitions):
		grads=[]

		beta=self._beta

		for trans in transitions:
			x,a,r,x_new=trans

			# print(x,a,r,x_new)
			Q_x=self._get_check_state(x)
			is_terminal= x_new == None

			if is_terminal:
				g = r-Q_x[a]
			else:
				Q_x_new=self._get_check_state(x_new)
				g = r+beta*np.max(Q_x_new)-Q_x[a]

			#gradient is of (y - Q(x,a;theta)), hence the -ve
			grads.append(-g)

		return grads

	"""
	=======Optional Methods======
	"""
	def get_opt_policy(self):
		return {s:np.argmax(self._get_check_state(s)) for s in self._Q}


	def save_data(self, fpath):
		if os.path.exists(fpath):
			warnings.warn('Overwriting existing model data!')

		if not '.pkl' in fpath:
			fpath+='.pkl'
		
		with open(fpath, 'wb') as f:
			pickle.dump(self._Q, f)

	
	def load_data(self, fpath):
		if not '.pkl' in fpath:
			fpath+='.pkl'
		
		with open(fpath, 'rb') as f:
			self._Q=pickle.load(f)

