import equinox as eqx
import sys
import logging
from typing import List

sys.path.append("..")
import copy
import pickle as pkl
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.linalg import block_diag

import simulation_class.model as model

logging.basicConfig(level="INFO")


class MC_PILCO(eqx.Module):
    """
    Monte-Carlo Probabilistic Inference for Learning COntrol
    """

    T_sampling: int
    state_dim: int
    input_dim: int

    def __init__(
        self,
        T_sampling,
        state_dim: int,
        input_dim: int,
        f_sim,
        f_model_learning,
        model_learning_par,
        f_rand_exploration_policy,
        rand_exploration_policy_par,
        f_control_policy,
        control_policy_par,
        f_cost_function,
        cost_function_par,
    ):
        super(MC_PILCO, self).__init__()
        # model parameters
        self.T_sampling = T_sampling  # sampling time

        self.state_dim: int = state_dim  # state dimension
        self.input_dim: int = input_dim  # input dimension

        # get the simulated system
        logging.info("\n\nGet the system...")
        self.system = model.Model(f_sim)  # ODE-simulated system

        if std_meas_noise is None:
            std_meas_noise = np.zeros(state_dim)
        self.std_meas_noise = std_meas_noise  # measurement noise

        # get the model learning object
        logging.info("\n\nGet the learning object...")
        self.model_learning = f_model_learning(**model_learning_par)

        # get the get the random explorataion policy object
        logging.info("\n\nGet the exploration policy...")
        self.rand_exploration_policy = f_rand_exploration_policy(
            **rand_exploration_policy_par
        )

        # get the initialize the policy parameters and get the policy object
        logging.info("\n\nGet the control policy...")
        self.control_policy = f_control_policy(**control_policy_par)

        logging.info("\n\nGet the cost function...")
        self.cost_function = f_cost_function(**cost_function_par)

        # state and input samples hystory list
        self.state_samples_history: List = []
        self.input_samples_history: List = []
        self.noiseless_states_history: List = []

        # initialize num_data_collection
        self.num_data_collection = 0

        # create log file dictionary
        self.log_path = log_path
        if self.log_path is not None:
            self.log_dict = {}

    def reinforce(
        self,
        initial_state,
        initial_state_var,
        T_exploration,
        T_control,
        num_trials,
        model_optimization_opt_list,
        policy_optimization_dict,
        num_explorations=1,
        flg_init_uniform=False,
        init_up_bound=None,
        init_low_bound=None,
        flg_init_multi_gauss=False,
        random_initial_state=True,
        loaded_model=False,
    ):
        """
        Model learning + policy learning method
        """
        # get initial data
        if not loaded_model:
            logging.info(
                "\n\n\n\n----------------- INITIAL EXPLORATIONS -----------------"
            )
            # perform 'num_explorations' interactions with the system to learn initial model
            for expl_index in range(0, num_explorations):
                logging.info("\nEXPLORATION # " + str(expl_index))
                if random_initial_state == True:  # initial state randomly sampled
                    if flg_init_uniform == True:  # uniform initial distribution
                        x0 = np.random.uniform(init_low_bound, init_up_bound)
                    elif (
                        flg_init_multi_gauss == True
                    ):  # multimodal gaussians initial distribution
                        num_init = np.random.randint(initial_state.shape[0])
                        x0 = np.random.normal(
                            initial_state[num_init, :],
                            np.sqrt(initial_state_var[num_init, :]),
                        )
                    else:  # gaussian initial distribution
                        x0 = np.random.normal(initial_state, np.sqrt(initial_state_var))
                else:  # deterministic initial state
                    x0 = initial_state

                # interact with the system
                self.get_data_from_system(
                    initial_state=x0,
                    T_exploration=T_exploration,
                    flg_exploration=True,  # exploration interaction
                    trial_index=expl_index,
                )
            cost_trial_list = []
            std_cost_trial_list = []
            parameters_trial_list = []
            particles_states_list = []
            particles_inputs_list = []

            first_trial_index = num_explorations - 1
            last_trial_index = num_trials + num_explorations - 1

        else:
            cost_trial_list = self.log_dict["cost_trial_list"]
            std_cost_trial_list = self.log_dict["std_cost_trial_list"]
            parameters_trial_list = self.log_dict["parameters_trial_list"]
            particles_states_list = self.log_dict["particles_states_list"]
            particles_inputs_list = self.log_dict["particles_inputs_list"]

            num_past_trials = len(self.state_samples_history)
            first_trial_index = num_past_trials - 1
            last_trial_index = num_trials + num_past_trials - 1

        # reinforce the model and the policy
        for trial_index in range(first_trial_index, last_trial_index):
            logging.info(
                "\n\n\n\n----------------- TRIAL "
                + str(trial_index)
                + " -----------------"
            )
            # train GPs on observed interaction data
            logging.info("\n\n----- REINFORCE THE MODEL -----")
            self.model_learning.reinforce_model(
                optimization_opt_list=model_optimization_opt_list
            )

            logging.info("\n\n----- REINFORCE THE POLICY -----")
            self.model_learning.set_eval_mode()

            # update the policy based on particle-simulation with the learned model
            (
                cost_list,
                std_cost_list,
                particles_states,
                particles_inputs,
            ) = self.reinforce_policy(
                T_control=T_control,
                particles_initial_state_mean=particles_initial_state_mean,
                particles_initial_state_var=particles_initial_state_var,
                flg_particles_init_uniform=flg_init_uniform,
                particles_init_up_bound=particles_init_up_bound,
                particles_init_low_bound=particles_init_low_bound,
                flg_particles_init_multi_gauss=flg_init_multi_gauss,
                trial_index=trial_index,
                **policy_optimization_dict
            )

            # save cost components
            cost_trial_list.append(cost_list)
            std_cost_trial_list.append(std_cost_list)
            particles_states_list.append(particles_states)
            particles_inputs_list.append(particles_inputs)
            parameters_trial_list.append(
                copy.deepcopy(self.control_policy.state_dict())
            )

            if self.log_path is not None:
                logging.info("Save log file...")
                self.log_dict["cost_trial_list"] = cost_trial_list
                self.log_dict["std_cost_trial_list"] = std_cost_trial_list
                self.log_dict["parameters_trial_list"] = parameters_trial_list
                self.log_dict["particles_states_list"] = particles_states_list
                self.log_dict["particles_inputs_list"] = particles_inputs_list
                pkl.dump(self.log_dict, open(self.log_path + "/log.pkl", "wb"))
            self.model_learning.set_training_mode()

            # test policy
            if random_initial_state == True:
                if flg_init_uniform == True:
                    x0 = np.random.uniform(init_low_bound, init_up_bound)
                elif flg_init_multi_gauss == True:
                    num_init = np.random.randint(initial_state.shape[0])
                    x0 = np.random.normal(
                        initial_state[num_init, :],
                        np.sqrt(initial_state_var[num_init, :]),
                    )
                else:
                    x0 = np.random.normal(initial_state, np.sqrt(initial_state_var))
            else:
                x0 = initial_state

            logging.info("\n\n----- APPLY THE CONTROL POLICY -----")
            # interact with the system
            self.get_data_from_system(
                initial_state=x0,
                T_exploration=T_control,
                flg_exploration=False,  # control policy interaction
                trial_index=trial_index + 1,
            )

        return cost_trial_list, particles_states_list, particles_inputs_list

    def rollout(self, data_collection_index, T_rollout=None, particle_pred=False):
        """
        Performs rollout of the data_collection_index trajectory
        """
        # check T_rollout
        if T_rollout is None:
            T_rollout = self.state_samples_history[data_collection_index].shape[0]
        # get initial state
        current_state_tc = torch.tensor(
            self.state_samples_history[data_collection_index][0:1, :],
            dtype=self.dtype,
            device=self.device,
        )
        # input trajectory as tensor
        inputs_trajectory_tc = torch.tensor(
            self.input_samples_history[data_collection_index],
            dtype=self.dtype,
            device=self.device,
        )
        # allocate the space for the rollout trajectory
        rollout_trj = torch.zeros(
            [T_rollout, self.state_dim], dtype=self.dtype, device=self.device
        )
        rollout_trj[0:1, :] = current_state_tc
        # simulate system evolution for 'T_rollout' steps
        for t in range(1, T_rollout):
            # get next state
            rollout_trj[t : t + 1, :], _, _ = self.model_learning.get_next_state(
                current_state=rollout_trj[t - 1 : t, :],
                current_input=inputs_trajectory_tc[t - 1 : t, :],
                particle_pred=particle_pred,
            )
        return rollout_trj.detach().cpu().numpy()

    def reinforce_policy(
        self,
        T_control,
        num_particles,
        trial_index,
        particles_initial_state_mean,
        particles_initial_state_var,
        flg_particles_init_uniform,
        particles_init_up_bound,
        particles_init_low_bound,
        flg_particles_init_multi_gauss,
        opt_steps_list,
        lr_list,
        f_optimizer,
        num_step_print=10,
        policy_reinit_dict=None,
        p_dropout_list=None,
        std_cost_filt_order=None,
        std_cost_filt_cutoff=None,
        max_std_cost=None,
        alpha_cost=0.99,
        alpha_input=0.99,
        alpha_diff_cost=0.99,
        lr_reduction_ratio=0.5,
        lr_min=0.001,
        p_drop_reduction=0.0,
        min_diff_cost=0.1,
        num_min_diff_cost=200,
        min_step=np.inf,
    ):
        """
        Improve the policy parameters with MC optimization
        """

        # init cost variables
        control_horizon = int(T_control / self.T_sampling)
        num_opt_steps = opt_steps_list[trial_index]
        cost_list = torch.zeros(num_opt_steps, device=self.device, dtype=self.dtype)
        std_cost_list = torch.zeros(num_opt_steps, device=self.device, dtype=self.dtype)
        previous_cost = 0.0
        current_min_step = min_step
        reinit_counter = 0

        # check dropout parameters
        if p_dropout_list is None:
            p_dropout = 0.0
        else:
            p_dropout = p_dropout_list[trial_index]
            logging.info("\nDROPOUT ACTIVE:")
            logging.info("p_dropout:", p_dropout)
        p_dropout_applied = p_dropout
        flg_drop = False

        # initilize the SE filter for monitoring cost improvement
        with torch.no_grad():
            num_attempts = 0
            flg_nan = True
            # repeat 'apply_policy' if nan is obtained
            while num_attempts < 10 and flg_nan:
                # apply the policy in simulation
                states_sequence_NODROP, inputs_sequence_NODROP = self.apply_policy(
                    particles_initial_state_mean=particles_initial_state_mean,
                    particles_initial_state_var=particles_initial_state_var,
                    flg_particles_init_uniform=flg_particles_init_uniform,
                    flg_particles_init_multi_gauss=flg_particles_init_multi_gauss,
                    particles_init_up_bound=particles_init_up_bound,
                    particles_init_low_bound=particles_init_low_bound,
                    num_particles=num_particles,
                    T_control=control_horizon,
                    p_dropout=p_dropout_applied,
                )
                # initial cost with no dropout applied
                cost_NODROP, std_cost_NODROP = self.cost_function(
                    states_sequence_NODROP, inputs_sequence_NODROP, trial_index
                )
                if torch.isnan(cost_NODROP):
                    num_attempts += 1
                    logging.info(
                        "\nSE filter initialization: Cost is NaN - reinit the policy"
                    )
                    self.control_policy.reinit(**policy_reinit_dict)
                else:
                    flg_nan = False

        # initilize filters
        ES1_diff_cost = torch.zeros(
            num_opt_steps + 1, device=self.device, dtype=self.dtype
        )
        ES2_diff_cost = 0.0
        diff_cost_ratio = torch.zeros(
            num_opt_steps + 1, device=self.device, dtype=self.dtype
        )
        cost_tm1 = cost_NODROP
        current_min_diff_cost = min_diff_cost

        # get the optimizer
        lr = lr_list[trial_index]  # list of learning rates for trial
        f_optim = eval(f_optimizer)
        optimizer = f_optim(p=self.control_policy.parameters(), lr=lr)

        # optimize the policy
        opt_step = 0
        opt_step_done = 0
        t_start = time.time()
        # max optimization steps = num_opt_steps
        while opt_step < num_opt_steps:
            # set the gradient to zero
            optimizer.zero_grad()

            num_attempts = 0
            flg_nan = True
            # repeat 'apply_policy' if nan is obtained
            while num_attempts < 10 and flg_nan:
                # apply the policy
                states_sequence, inputs_sequence = self.apply_policy(
                    particles_initial_state_mean=particles_initial_state_mean,
                    particles_initial_state_var=particles_initial_state_var,
                    flg_particles_init_uniform=flg_particles_init_uniform,
                    flg_particles_init_multi_gauss=flg_particles_init_multi_gauss,
                    particles_init_up_bound=particles_init_up_bound,
                    particles_init_low_bound=particles_init_low_bound,
                    num_particles=num_particles,
                    T_control=control_horizon,
                    p_dropout=p_dropout_applied,
                )
                # compute the expected cost
                cost, std_cost = self.cost_function(
                    states_sequence, inputs_sequence, trial_index
                )
                if torch.isnan(cost):
                    num_attempts += 1
                    logging.info("\nCost is NaN: try sampling again")
                else:
                    flg_nan = False

            # save current step's cost
            cost_list[opt_step] = cost.data.clone().detach()
            std_cost_list[opt_step] = std_cost.data.clone().detach()

            # update filters
            with torch.no_grad():
                # compute the mean of the diff cost
                ES1_diff_cost[opt_step + 1] = alpha_diff_cost * ES1_diff_cost[
                    opt_step
                ] + (1 - alpha_diff_cost) * (cost - cost_tm1)
                ES2_diff_cost = alpha_diff_cost * (
                    ES2_diff_cost
                    + (1 - alpha_diff_cost)
                    * ((cost - cost_tm1 - ES1_diff_cost[opt_step]) ** 2)
                )
                cost_tm1 = cost_list[opt_step]
                diff_cost_ratio[opt_step + 1] = alpha_diff_cost * diff_cost_ratio[
                    opt_step
                ] + (1 - alpha_diff_cost) * (
                    ES1_diff_cost[opt_step + 1] / (ES2_diff_cost.sqrt())
                )

            # compute the gradient and optimize the policy
            cost.backward(retain_graph=False)

            # updata parameters
            optimizer.step()

            # check improvement
            if opt_step % num_step_print == 0:
                t_stop = time.time()
                improvement = previous_cost - cost.data.cpu().numpy()
                previous_cost = cost.data.cpu().numpy()
                logging.info("\nOptimization step: ", opt_step)
                logging.info("cost: ", previous_cost)
                logging.info("cost improvement: ", improvement)
                logging.info("p_dropout_applied: ", p_dropout_applied)
                logging.info("current_min_diff_cost; ", current_min_diff_cost)
                logging.info("current_min_step: ", current_min_step)
                logging.info(
                    "diff_cost_ratio: ",
                    torch.abs(diff_cost_ratio[opt_step + 1]).cpu().numpy(),
                )
                logging.info("time elapsed: ", t_stop - t_start)
                t_start = time.time()

            # check learning rate and exit conditions
            if opt_step > current_min_step:
                if (
                    torch.sum(
                        torch.abs(
                            diff_cost_ratio[
                                opt_step + 1 - num_min_diff_cost : opt_step + 1
                            ]
                        )
                        < current_min_diff_cost
                    )
                    >= num_min_diff_cost
                ):
                    if lr > lr_min:
                        logging.info("Optimization_step:", opt_step)
                        logging.info("\nREDUCING THE LEARNING RATE:")
                        lr = max(lr * lr_reduction_ratio, lr_min)
                        logging.info("lr: ", lr)
                        current_min_diff_cost = max(current_min_diff_cost / 2, 0.01)
                        current_min_step = opt_step + num_min_diff_cost
                        optimizer = f_optim(p=self.control_policy.parameters(), lr=lr)
                        logging.info("\nREDUCING THE DROPOUT:")
                        p_dropout_applied = p_dropout_applied - p_drop_reduction
                        if p_dropout_applied < 0:
                            p_dropout_applied = 0.0
                        logging.info("p_dropout_applied: ", p_dropout_applied)
                    else:
                        logging.info(
                            "\nEXIT FROM OPTIMIZATION: diff_cost_ratio < min_diff_cost for num_min_diff_cost steps"
                        )
                        opt_step = num_opt_steps

            # increase step counter
            opt_step = opt_step + 1
            opt_step_done = opt_step_done + 1

            # reinit policy if NaN appeared
            if flg_nan:
                reinit_counter = reinit_counter + 1
                error_message = "Cost is NaN:"
                logging.info(
                    "\n"
                    + error_message
                    + " re-initialize control policy [attempt #"
                    + str(reinit_counter)
                    + "]"
                )
                self.control_policy.reinit(**policy_reinit_dict)
                # reset counter to 0
                opt_step = 0
                opt_step_done = 0
                current_min_step = min_step
                previous_cost = 0.0
                # re-init cost variables
                cost_list = torch.zeros(
                    num_opt_steps, device=self.device, dtype=self.dtype
                )
                std_cost_list = torch.zeros(
                    num_opt_steps, device=self.device, dtype=self.dtype
                )
                gradients_list = []
                states_sequence_std_list = torch.zeros(
                    [self.state_dim, num_opt_steps],
                    device=self.device,
                    dtype=self.dtype,
                )
                cost_list_NODROP = torch.zeros(
                    num_opt_steps, device=self.device, dtype=self.dtype
                )
                std_cost_list_NODROP = torch.zeros(
                    num_opt_steps, device=self.device, dtype=self.dtype
                )
                mean_states_list = torch.zeros(
                    [self.state_dim, num_opt_steps, control_horizon],
                    device=self.device,
                    dtype=self.dtype,
                )
                mean_inputs_list = torch.zeros(
                    [self.input_dim, num_opt_steps, control_horizon],
                    device=self.device,
                    dtype=self.dtype,
                )
                drop_list = []
                ES1_diff_cost = torch.zeros(
                    num_opt_steps + 1, device=self.device, dtype=self.dtype
                )
                diff_cost_ratio = torch.zeros(
                    num_opt_steps + 1, device=self.device, dtype=self.dtype
                )
                current_min_diff_cost = min_diff_cost
                # re-init the optimizer
                lr = lr_list[trial_index]
                f_optim = eval(f_optimizer)
                optimizer = f_optim(p=self.control_policy.parameters(), lr=lr)
                # Reinit dropout
                p_dropout_applied = p_dropout

        # move log variables to numpy
        cost_list = cost_list[0:opt_step_done].detach().cpu().numpy()
        std_cost_list = std_cost_list[0:opt_step_done].detach().cpu().numpy()

        return (
            cost_list,
            std_cost_list,
            states_sequence.detach().cpu().numpy(),
            inputs_sequence.detach().cpu().numpy(),
        )

    def apply_policy(
        self,
        particles_initial_state_mean,
        particles_initial_state_var,
        flg_particles_init_uniform,
        particles_init_up_bound,
        particles_init_low_bound,
        flg_particles_init_multi_gauss,
        num_particles,
        T_control,
        p_dropout=0.0,
    ):
        """
        Apply the policy in simulation to a batch of particles:
        """
        # initialize variables
        states_sequence_list = []
        inputs_sequence_list = []

        # get initial particles
        if flg_particles_init_uniform == True:
            # initial uniform distribution
            uniform_ub_particles = particles_init_up_bound.repeat(num_particles, 1)
            uniform_lb_particles = particles_init_low_bound.repeat(num_particles, 1)
            state_distribution = Uniform(uniform_lb_particles, uniform_ub_particles)
        elif flg_particles_init_multi_gauss == True:
            # initial multimodal Gaussian distribution
            indices = torch.randint(
                0, particles_initial_state_mean.shape[0], [num_particles]
            )
            initial_particles_mean = particles_initial_state_mean[indices, :]
            initial_particles_cov_mat = torch.stack(
                [torch.diag(particles_initial_state_var[i, :]) for i in indices]
            )
            state_distribution = MultivariateNormal(
                loc=initial_particles_mean, covariance_matrix=initial_particles_cov_mat
            )
        else:
            # initial Gaussian distribution
            initial_particles_mean = particles_initial_state_mean.repeat(
                num_particles, 1
            )
            initial_particles_cov_mat = torch.stack(
                [torch.diag(particles_initial_state_var)] * num_particles
            )
            state_distribution = MultivariateNormal(
                loc=initial_particles_mean, covariance_matrix=initial_particles_cov_mat
            )

        # sample particles at t=0 from initial state distribution
        states_sequence_list.append(state_distribution.rsample())

        # compute initial inputs
        inputs_sequence_list.append(
            self.control_policy(states_sequence_list[0], t=0, p_dropout=p_dropout)
        )

        for t in range(1, int(T_control)):
            # get next state mean and variance (given the states sampled and the inputs computed)
            particles, _, _ = self.model_learning.get_next_state(
                current_state=states_sequence_list[t - 1],
                current_input=inputs_sequence_list[t - 1],
            )
            states_sequence_list.append(particles)

            # compute next input
            inputs_sequence_list.append(
                self.control_policy(states_sequence_list[t], t=t, p_dropout=p_dropout)
            )

        # returns states/inputs trajectories
        return torch.stack(states_sequence_list), torch.stack(inputs_sequence_list)

    def get_data_from_system(
        self, initial_state, T_exploration, trial_index, flg_exploration=False
    ):
        """
        Apply exploration/control policy to the system and collect interaction data
        """
        # select the policy
        if flg_exploration:
            current_policy = self.rand_exploration_policy
        else:
            current_policy = self.control_policy

        # method for interacting with ODE-simulated system
        state_samples, input_samples, noiseless_samples = self.system.rollout(
            s0=initial_state,
            policy=current_policy.get_np_policy(),
            T=T_exploration,
            dt=self.T_sampling,
            noise=self.std_meas_noise,
        )
        self.state_samples_history.append(state_samples)
        self.input_samples_history.append(input_samples)
        self.noiseless_states_history.append(noiseless_samples)
        self.num_data_collection += 1
        # add data to model_learning object
        self.model_learning.add_data(
            new_state_samples=state_samples, new_input_samples=input_samples
        )
