# qlearning
An object-oriented, highly flexible implementation of Q-Learning, suitable for tabular and deep Q-networks.

## Code Structure ##
This repo provides two flexible classes for implementing a broad range of tabular and deep Q-learning models. 
* `qfunction.py` provides a class called `QFunction` which implements the Q-network.
* `qlearning.py` provides a class called `QLearner` which implements the Q-learning algorithm using a QFunction.

`QLearner` is a stateful representation of the Q-Learning algorithm, making generic method calls to an insance of `QFunction` which handles the particulars of updating the parameters of the Q-network.

## How to Use ##
See Examples for an example.
1. __setup the QFunction__: `qfunction.py` comes with a tabular  `QMat()` class, which can implement a sparse tabular Q-network, requiring only that the states are hashable objects, and there are finitely-many actions. Otherwise, you must derive your own class from QFunction and implement the required methods.
2. __setup the QLearner__: The default Q-netowrk is tabular (above) otherwise you must provide your own derived QFunction class. In addition to the learning parameters, etc., you must derive your own class and implement the adapter methods.
3. __insert the QLearner__: The QLearner gets an initial state, then learns via executing `act()`, `observe()`, and `update()` (or `update_batch()`). These must be executed in sequence within the environment in-order, or the learner will complain (loudly). 

## Design Philosophy ##
The stateful, OOP design of QLearner abstracts the Q-learning details from any specific problem, and interfaces with its environment via adapter methods. Some design principles we tried to embody here:
* __Transparency__: The individual components of the QLearning and QFunction models are in clearly-separated methods, each with a specific function.
* __Flexibility__: Only the absolutely necessary components have been provided. Everything else can be customized and, if necessary, be overriden. In addition, the Q-network's implementation is completely abstracted from its function, so different representations can be used with minimal changes.
* __Generality__: This software reflects a unified view of the mathematical Q-Learning algorithm as an online regression problem (e.g. with a loss function, an optimizer, etc). This view unifies traditional Q-Learning as well as modern Deep Q-Learning methods.
* __Portability__: The same Q-learning model can be used in different environments, modifiying only the adapter methods.

## Example ##
I have used this framework to perform tabular Q-Learning to play the game FlappyBird. A trained version of this model is in the Examples folder, along with examples of each of the steps above. This model draws inspriation from other excellent implementations by [yenchenlin](https://github.com/yenchenlin/DeepLearningFlappyBird/) and [chncynh](https://github.com/chncyhn/flappybird-qlearning-bot/). My model underperforms theirs in-terms of score, but there is evidence that I just didn't train the model for long-enough or find the right hyperparameters.

![pic](https://github.com/jamesvuc/qlearning/blob/master/Examples/FlapPyBird-master/FlappyGif.gif "Flappy Bird Result")

## Next Steps ##
1. Provide a deep Q-Learning example.
2. Incorporate more general TD(lambda) learning models. 
3. Support online learning (e.g. SARSA).
