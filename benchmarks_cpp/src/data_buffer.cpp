#include <cmath>
#include "replay_buffer.hpp"
#include "data_buffer.hpp"
#include "serialize.hpp"
#include "debug.hpp"

DataBuffer::DataBuffer(int capacity, int n_steps, float gamma, float initial_priority, int n_children)
    : past_actions(n_steps), past_rewards(n_steps), past_dones(n_steps), device(ReplayBuffer::getDevice()) {

    // Store the data buffer's parameters.
    this->capacity = capacity;
    this->n_steps = n_steps;
    this->gamma = gamma;

    // Torch tensors storing all the buffer's data.
    this->actions = torch::zeros({capacity}, at::kInt).to(this->device);
    this->rewards = torch::zeros({capacity}, at::kFloat).to(this->device);
    this->dones = torch::zeros({capacity}, at::kBool).to(this->device);

    // The priorities associated with all experiences in the replay buffer.
    this->priorities = std::make_unique<PriorityTree>(capacity, initial_priority, n_children);

    // The index of the next datum to add in the buffer.
    this->current_id = 0;
}

void DataBuffer::append(const ExperienceTuple &experience_tuple) {

    Experience experience = Experience(experience_tuple);

    // Update the returns of last experiences.
    for (std::uint64_t i = 0; i < this->past_rewards.size(); i++) {
        this->past_rewards[i] += std::pow(this->gamma, i + 1) * experience.reward;
    }

    // Add the reward, action and done to their respective queues.
    this->past_rewards.push_front(experience.reward);
    this->past_actions.push_front(experience.action);
    this->past_dones.push_front(experience.done);

    // Add new data to the buffer.
    if (experience.done == true) {

        // If the current episode has ended, keep track of all valid data.
        while (this->past_rewards.size() != 0) {
            this->addDatum(this->past_actions.back(), this->past_rewards.back(), this->past_dones[0]);
            this->past_actions.pop_back();
            this->past_rewards.pop_back();
        }

        // Then, clear the queues of past reward, actions, and dones.
        this->past_rewards.clear();
        this->past_actions.clear();
        this->past_dones.clear();

    } else if (static_cast<int>(this->past_rewards.size()) == this->n_steps) {

        // If the current episode has not ended, but the queues are full, then keep track of next valid datum.
        this->addDatum(this->past_actions.back(), this->past_rewards.back(), this->past_dones[0]);
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> DataBuffer::operator[](torch::Tensor &indices) {
    if (this->current_id >= this->capacity) {
        indices = torch::remainder(indices + this->current_id, this->capacity);
    }
    return std::make_tuple(this->actions.index({indices}), this->rewards.index({indices}), this->dones.index({indices}));
}

int DataBuffer::size() {
    return std::min(this->current_id, this->capacity);
}

void DataBuffer::clear() {
    this->past_actions.clear();
    this->past_rewards.clear();
    this->past_dones.clear();
    this->actions = torch::zeros({capacity}, at::kInt).to(this->device);
    this->rewards = torch::zeros({capacity}, at::kFloat).to(this->device);
    this->dones = torch::zeros({capacity}, at::kBool).to(this->device);
    this->priorities->clear();
    this->current_id = 0;
}

void DataBuffer::addDatum(int action, float reward, bool done) {
    int index = this->current_id % this->capacity;
    this->actions.index_put_({index}, action);
    this->rewards.index_put_({index}, reward);
    this->dones.index_put_({index}, done);
    this->priorities->append(this->priorities->max());
    this->current_id += 1;
}

std::unique_ptr<PriorityTree> &DataBuffer::getPriorities() {
    return this->priorities;
}

void DataBuffer::load(std::istream &checkpoint) {

    // Load the data buffer from the checkpoint.
    this->capacity = serialize::load_value<int>(checkpoint);
    this->n_steps = serialize::load_value<int>(checkpoint);
    this->gamma = serialize::load_value<float>(checkpoint);
    this->past_actions.load(checkpoint);
    this->past_rewards.load(checkpoint);
    this->past_dones.load(checkpoint);
    this->actions = serialize::load_tensor<int>(checkpoint);
    this->rewards = serialize::load_tensor<float>(checkpoint);
    this->dones = serialize::load_tensor<bool>(checkpoint);
    this->priorities->load(checkpoint);
    this->current_id = serialize::load_value<int>(checkpoint);
}

void DataBuffer::save(std::ostream &checkpoint) {

    // Save the data buffer in the checkpoint.
    serialize::save_value(this->capacity, checkpoint);
    serialize::save_value(this->n_steps, checkpoint);
    serialize::save_value(this->gamma, checkpoint);
    this->past_actions.save(checkpoint);
    this->past_rewards.save(checkpoint);
    this->past_dones.save(checkpoint);
    serialize::save_tensor<int>(this->actions, checkpoint);
    serialize::save_tensor<float>(this->rewards, checkpoint);
    serialize::save_tensor<bool>(this->dones, checkpoint);
    this->priorities->save(checkpoint);
    serialize::save_value(this->current_id, checkpoint);
}

void DataBuffer::print(bool verbose, const std::string &prefix) {

    // Display the most important information about the data buffer.
    std::cout << "DataBuffer[capacity: " << this->capacity << ", n_steps: " << this->n_steps
              << ", gamma: " << this->gamma << ", current_id: " << this->current_id << "]" << std::endl;

    // Display optional information about the data buffer.
    if (verbose == true) {
        std::cout << prefix << " #-> past_actions = ";
        this->past_actions.print();
        std::cout << prefix << " #-> past_rewards = ";
        this->past_rewards.print();
        std::cout << prefix << " #-> past_dones = ";
        this->past_dones.print();
        std::cout << prefix << " #-> actions = ";
        debug::print_tensor<int>(this->actions, 10);
        std::cout << prefix << " #-> rewards = ";
        debug::print_tensor<float>(this->rewards, 10);
        std::cout << prefix << " #-> dones = ";
        debug::print_tensor<bool>(this->dones, 10);
        std::cout << prefix  << " #-> priority_tree = ";
        this->priorities->print(verbose, prefix + "     ");
    }
}
