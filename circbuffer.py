import random
from collections import deque

class CircularBuffer:
	"""
	A fixed-size buffer that supports O(1) insert, remove, O(S) random sampling 
	where S is the sample size, and O(C) construction and reset where C is the capacity.
	It is implemented as an array-based circular queue.


	This should be used when you expect to be inserting and sampling often, as in 
	a replay memory buffer for RL.

	"""
	def __init__(self, size):
		self._capacity=size

		#initialize the buffer
		self._buff=[None for i in range(self._capacity)]

		#iniditalize indices
		self._front=self._capacity-1# always points to the front item
		self._back=self._capacity-1# always points to the next available space

		#initialize size
		self._size=0

	#add a new element to the queue
	def put(self, x):
		#the buffer is full
		if self.full():
			#create a space (also decreases size)
			self._pop()

		self._buff[self._back]=x
		self._back = (self._back - 1) % self._capacity

		self._size+=1

	#helper method for removing from the queue
	def _pop(self):
		if not self.empty():
			#store the contents
			tmp=self._buff[self._front]
			#reset the value
			self._buff[self._front]=None
			#decrement the pointer
			self._front = (self._front - 1) % self._capacity

			self._size-=1

			return tmp

	def full(self):
		return ( (self._front - self._back) % self._capacity == 0 )  and ( self._size>0 ) 

	def empty(self):
		return self._size == 0

	#get items at idxs
	def sample(self, size=10):
		if self.empty():
			print("Circular Buffer is empty!")
			# raise ValueError("Circular Buffer is empty!")
			return []

		idxs=random.choices(range(self._size), k=size)
		#could vectorize this with numpy. not sure if it would be faster...
		if self.full():
			return [self._buff[idx] for idx in idxs]
		else:
			return [self._buff[(idx+self._back+1)% self._capacity] for idx in idxs]

	def display(self):
		print(self._buff)

	#you could theoretically just put _front=_back=_capacity-1 and set size=0.
	def reset(self):
		self.__init__(self._capacity)


class SlowBuffer:
	"""
	A buffer with the same interface using a double-ended queue for comparison. 
	Considerably simpler to implement, but this has O(C) sampling complexity since
	random.sample(deque()) converts whole the linked-list to a regular strucutre with 
	random access before sampling.
	"""
	def __init__(self, size=10):
		self._capacity=size
		self._buff=deque()
		self._size=0

	def put(self, x):
		if self._size > self._capacity:
			self._pop()

		self._size+=1
		self._buff.appendleft(x)

	def _pop(self):
		self._buff.pop()
		self._size-=1

	def sample(self, size=10):
		# return np.random.choice(self._buff, size=size)
		if size > self._size or size == -1:
			return list(self._buff)
		return random.sample(self._buff, size)

	def reset(self):
		self._buff.clear()
		self._size=0

	def display(self):
		print(self._buff)

if __name__=='__main__':
	"""
	A quick unit test/speed comparison
	"""

	BSIZE=50000
	N_ITERS=int(1e6)
	SAMP_FREQ=5
	SAMP_SIZE=32

	import datetime as dt

	# ======= SLOW =======
	t1=dt.datetime.now()

	B=Buffer(BSIZE)
	for i in range(N_ITERS):
		B.put((i,1))
		if i % SAMP_FREQ == 0 :
			B.sample(size=SAMP_SIZE)

	t1=dt.datetime.now()-t1

	# ======= FAST ======

	t2=dt.datetime.now()

	cb=CircularBuffer(size=BSIZE)
	for i in range(N_ITERS):
		cb.put(i)
		if i % SAMP_FREQ == 0:
			cb.sample(size=SAMP_SIZE)

	t2=dt.datetime.now()-t2

	# ======= RESULTS =====

	print('deque =',t1)
	print('circular queue =',t2)

