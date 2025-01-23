#ifndef DATA_BUFFER_HPP
#define DATA_BUFFER_HPP

#include "agents/memory/priority_tree.hpp"
#include "agents/memory/experience.hpp"
#include "helpers/deque.hpp"

namespace relab::agents::memory::impl {

    using namespace relab::helpers;

    /**
     * A buffer allowing for storage and retrieval of experience datum (i.e., action, reward, done, and priority).
     */
    class DataBuffer {

    private:

        // Store the data buffer's parameters.
        int capacity;
        int n_steps;
        float gamma;

        // Queues keeping track of past actions, cumulated rewards, and dones.
        Deque<int> past_actions;
        Deque<float> past_rewards;
        Deque<bool> past_dones;

        // Torch tensors storing all the buffer's data.
        torch::Device device;
        torch::Tensor actions;
        torch::Tensor rewards;
        torch::Tensor dones;

        // The priorities associated with all experiences in the replay buffer.
        std::unique_ptr<PriorityTree> priorities;

        // The index of the next datum to add in the buffer.
        int current_id;

    public:

        /**
         * Create a data buffer.
         * @param capacity the number of experiences the buffer can store
         * @param n_steps the number of steps for which rewards are accumulated in multistep Q-learning
         * @param gamma the discount factor
         * @param initial_priority the initial priority given to first elements
         * @param n_children the number of children each node has
         */
        DataBuffer(int capacity, int n_steps, float gamma, float initial_priority, int n_children);

        /**
         * Add the datum of the next experience to the buffer.
         * @param experience the experience whose datum must be added to the buffer
         */
        void append(const Experience &experience);

        /**
         * Retrieve the data of the experiences whose indices are passed as parameters.
         * @param indices the indices of the experiences whose data must be retrieved
         * @return the data (i.e., action at time t, n-steps return at time t, and done at time t + n_steps)
         */
        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> operator[](torch::Tensor &indices);

        /**
         * Retrieve the number of experiences stored in the buffer.
         * @return the number of experiences stored in the buffer
         */
        int size();

        /**
         * Empty the data buffer.
         */
        void clear();

        /**
         * Add a datum to the buffer.
         * @param action the action at time t
         * @param reward the n-steps reward
         * @param done the done at time t + n_steps
         */
        void addDatum(int action, float reward, bool done);

        /**
         * Retrieve the priority tree.
         * @return the priority tree
         */
        std::unique_ptr<PriorityTree> &getPriorities();

        /*
         * Load the data buffer from the checkpoint.
         * @param checkpoint a stream reading from the checkpoint file
         */
        void load(std::istream &checkpoint);

        /*
         * Save the data buffer in the checkpoint.
         * @param checkpoint a stream writing into the checkpoint file
         */
        void save(std::ostream &checkpoint);

        /**
         * Print the data buffer on the standard output.
         * @param verbose true if the full data buffer should be displayed, false otherwise
         * @param prefix the prefix to add an front of the optional information
         */
        void print(bool verbose=false, const std::string &prefix="");
    };
}

namespace relab::agents::memory {
    using impl::DataBuffer;
}

#endif //DATA_BUFFER_HPP
