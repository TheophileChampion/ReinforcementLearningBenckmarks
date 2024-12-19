from benchmarks import benchmarks
from benchmarks.agents.memory.cpp import ReplayBuffer


class FastReplayBuffer:
    """
    Class implementing a replay buffer with support for prioritization [1] and multistep Q-learning [2] from:

    [1] Tom Schaul. Prioritized experience replay. arXiv preprint arXiv:1511.05952, 2015.
    [2] Richard S Sutton. Learning to predict by the methods of temporal differences. Machine learning, 3:9–44, 1988.
    """

    def __init__(self, capacity=10000, batch_size=32, frame_skip=None, stack_size=None, p_args=None, m_args=None):
        """
        Create a replay buffer.
        :param capacity: the number of experience the buffer can store
        :param batch_size: the size of the batch to sample
        :param frame_skip: the number of times each action is repeated in the environment, if None use the configuration
        :param stack_size: the number of stacked frame in each observation, if None use the configuration
        :param p_args: the prioritization arguments (None for no prioritization) composed of:
            - initial_priority: the maximum experience priority given to new transitions
            - omega: the prioritization exponent
            - omega_is: the important sampling exponent
            - n_children: the maximum number of children each node of the priority-tree can have
        :param m_args: the multistep arguments (None for no multistep) composed of:
            - n_steps: the number of steps for which rewards are accumulated in multistep Q-learning
            - gamma: the discount factor
        """
        stack_size = benchmarks.config("stack_size") if stack_size is None else stack_size
        frame_skip = benchmarks.config("frame_skip") if frame_skip is None else frame_skip
        p_args = {} if p_args is None else p_args
        m_args = {} if m_args is None else m_args
        self.buffer = ReplayBuffer(
            capacity=capacity, batch_size=batch_size, frame_skip=frame_skip, stack_size=stack_size,
            p_args=p_args, m_args=m_args
        )

    def append(self, experience):
        """
        Add a new experience to the buffer.
        :param experience: the experience to add
        """
        self.buffer.append(experience)

    def sample(self):
        """
        Sample a batch from the replay buffer.
        :return: observations, actions, rewards, done, next_observations
        where:
        - observations: the batch of observations
        - actions: the actions performed
        - rewards: the rewards received
        - done: whether the environment stop after performing the actions
        - next_observations: the observations received after performing the actions
        """
        return self.buffer.sample()

    def report(self, loss):
        """
        Report the loss associated with all the transitions of the previous batch.
        :param loss: the loss of all previous transitions
        :return: the new loss
        """
        return self.buffer.report(loss)

    def get_experiences(self, indices):
        """
        Retrieve the experiences whose indices are passed as parameters.
        :param indices: the experience indices
        :return: the experiences
        """
        return self.buffer.get_experiences(indices)

    def clear(self):
        """
        Empty the replay buffer.
        """
        self.buffer.clear()

    def __len__(self):
        """
        Retrieve the number of elements in the buffer.
        :return: the number of elements contained in the replay buffer
        """
        return self.buffer.length()

    def is_prioritized(self):
        """
        Retrieve a boolean indicating whether the replay buffer is prioritized.
        :return: true if the replay buffer is prioritized, false otherwise
        """
        return self.buffer.is_prioritized()

    def get_last_indices(self):
        """
        Retrieves the last sampled indices.
        :return: the last sampled indices
        """
        return self.buffer.get_last_indices()

    def get_priority(self, index):
        """
        Retrieves the priority at the provided index.
        :param index: the index
        :return: the priority
        """
        return self.buffer.get_priority(index)