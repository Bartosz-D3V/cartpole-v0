# OpenAI Gym's CartPole resolved with Q-Learning

CartPole problem resolved with simple Q-Learning algorithm.

Because the implementation is a simple Q-learning algorithm, the optimization is not great compared to alternative solutions
(like DQN). It takes about ~230 iterations to converge.

Function convergence is defined as when the last 100 episodes yields average score of at least 195.

## Prerequisites
* Python 3.7 installed
* Pipenv installed

## Installation
`pipenv install`

## Running
`pipenv shell`

`pipenv run python -m main` // or your favourite Python IDE

## License
MIT
