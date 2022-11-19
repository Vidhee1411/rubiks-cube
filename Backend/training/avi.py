from utils import data_utils, nnet_utils, env_utils, search_utils, misc_utils
from typing import Dict, List, Tuple, Any

from environments.environment_abstract import Environment, State
from search_methods.astar import AStar, Node
from search_methods.gbfs import GBFS, gbfs_test

import torch
from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
import os
import pickle

from argparse import ArgumentParser
import numpy as np
import time

import sys
import shutil


def parse_arguments(parser: ArgumentParser) -> Dict[str, Any]:
    # Environment
    parser.add_argument('--env', type=str, required=True, help="Environment")

    # Debug
    parser.add_argument('--debug', action='store_true', default=False, help="")

    # Gradient Descent
    parser.add_argument('--lr', type=float, default=0.001, help="Initial learning rate")
    parser.add_argument('--lr_d', type=float, default=0.9999993, help="Learning rate decay for every iteration. "
                                                                      "Learning rate is decayed according to: "
                                                                      "lr * (lr_d ^ itr)")

    # Training
    parser.add_argument('--batch_size', type=int, default=100, help="Batch size")
    parser.add_argument('--frac_search', type=float, default=0.5, help="Fraction for searching")
    parser.add_argument('--subgoal_start', type=int, default=0, help="Subgoal at which to start")

    # Update
    parser.add_argument('--astar_weight', type=float, default=0.5, help="Weight on path cost when solving with A*")
    parser.add_argument('--loss_thresh', type=float, default=0.1, help="When the loss falls below this value, "
                                                                       "the target network is updated to the current "
                                                                       "network.")
    parser.add_argument('--states_per_update', type=int, default=100000, help="How many states to train on before "
                                                                              "checking if target network should be "
                                                                              "updated")
    parser.add_argument('--epochs_per_update', type=int, default=1, help="How many epochs to train for. "
                                                                         "Making this greater than 1 could increase "
                                                                         "risk of overfitting, however, one can train "
                                                                         "for more iterations without having to "
                                                                         "generate more data.")
    parser.add_argument('--update_search_steps', type=int, default=100, help="")
    parser.add_argument('--update_nnet_batch_size', type=int, default=10000, help="Batch size of each nnet used for "
                                                                                  "each process update. "
                                                                                  "Make smaller if running out of "
                                                                                  "memory.")
    parser.add_argument('--eps', type=float, default=0.1, help="For epsilon greedy q-learning")

    # Testing
    parser.add_argument('--num_test', type=int, default=1000, help="Number of test states.")
    parser.add_argument('--astar_steps_incr', type=int, default=10, help="Number of steps for A* to increment every "
                                                                         "udpate")
    parser.add_argument('--max_astar_steps', type=int, default=1000, help="Maximum number of steps for A*")

    # data
    parser.add_argument('--back_max', type=int, required=True, help="Maximum number of backwards steps from goal")

    # model
    parser.add_argument('--nnet_name', type=str, required=True, help="Name of neural network")
    parser.add_argument('--save_dir', type=str, default="saved_models", help="Director to which to save model")

    # parse arguments
    args = parser.parse_args()

    args_dict: Dict[str, Any] = vars(args)

    # make save directory
    args_dict['model_dir'] = "%s/%s/" % (args_dict['save_dir'], args_dict['nnet_name'])
    if not os.path.exists(args_dict['model_dir']):
        os.makedirs(args_dict['model_dir'])

    args_dict["output_save_loc"] = "%s/output.txt" % args_dict['model_dir']

    # save args
    args_save_loc = "%s/args.pkl" % args_dict['model_dir']
    print("Saving arguments to %s" % args_save_loc)
    with open(args_save_loc, "wb") as f:
        pickle.dump(args, f, protocol=-1)

    print("Batch size: %i" % args_dict['batch_size'])

    assert args_dict['frac_search'] < 1.0, "frac_search must be less than 1"

    return args_dict


def copy_files(src_dir: str, dest_dir: str):
    src_files: List[str] = os.listdir(src_dir)
    for file_name in src_files:
        full_file_name: str = os.path.join(src_dir, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, dest_dir)


def get_subgoal_num(model_dir: str, subgoal_num_start) -> int:
    subgoal_num_file: str = "%s/subgoal_num.pkl" % model_dir
    if os.path.isfile(subgoal_num_file):
        subgoal_num: int = pickle.load(open(subgoal_num_file, "rb"))
    else:
        subgoal_num: int = subgoal_num_start

    return subgoal_num


def load_nnet(nnet_dir: str, env: Environment) -> Tuple[nn.Module, int, int]:
    nnet_file: str = "%s/model_state_dict.pt" % nnet_dir
    if os.path.isfile(nnet_file):
        nnet = nnet_utils.load_nnet(nnet_file, env.get_nnet())
        itr: int = pickle.load(open("%s/train_itr.pkl" % nnet_dir, "rb"))
        update_num: int = pickle.load(open("%s/update_num.pkl" % nnet_dir, "rb"))
    else:
        nnet: nn.Module = env.get_nnet()
        itr: int = 0
        update_num: int = 0

    return nnet, itr, update_num


def get_itr_tot(model_dir: str) -> int:
    itr_tot_file: str = "%s/itr_tot.pkl" % model_dir
    if os.path.isfile(itr_tot_file):
        itr_tot: int = pickle.load(open(itr_tot_file, "rb"))
    else:
        itr_tot: int = 0

    return itr_tot


def reach_subgoal(states: List[State], weight: float, env: Environment, heuristic_fn) -> List[State]:
    astar = AStar(states, env, heuristic_fn, weights=[weight] * len(states))
    while not min(astar.has_found_goal()):
        astar.step(heuristic_fn, 1)

    # set states to be the subgoals that were reached
    states = []
    for state_idx in range(len(astar.instances)):
        state_goal = astar.get_cheapest_goal_node(state_idx).state
        states.append(state_goal)

    return states


def scrambled_to_subgoal(subgoal_num_start: int, subgoal_num: int, states: List[State], weight: float,
                         envs_sg: List[Environment], model_dir: str, device, on_gpu: bool,
                         nnet_batch_size) -> List[State]:
    for subgoal_num_prev in range(subgoal_num_start, subgoal_num + 1):
        start_time = time.time()
        env_sg_prev = envs_sg[subgoal_num_prev]
        nnet_prev_dir = "%s/%s/current/" % (model_dir, subgoal_num_prev)
        heuristic_fn_sg_prev = nnet_utils.load_heuristic_fn(nnet_prev_dir, device, on_gpu,
                                                            env_sg_prev.get_nnet(), env_sg_prev,
                                                            clip_zero=False,
                                                            batch_size=nnet_batch_size)

        states = reach_subgoal(states, weight, env_sg_prev, heuristic_fn_sg_prev)
        print("Time to reach subgoal %i: %f" % (subgoal_num_prev, time.time() - start_time))

    return states


def try_to_reach_subgoal_astar(states: List[State], weight: float, env: Environment, heuristic_fn, num_steps: int):
    astar = AStar(states, env, heuristic_fn, weights=[weight] * len(states))
    for _ in range(num_steps):
        if not min(astar.has_found_goal()):
            astar.step(heuristic_fn, 1, verbose=False)

    nodes_popped: List[List[Node]] = astar.get_popped_nodes()
    nodes_popped_flat: List[Node]
    nodes_popped_flat, _ = misc_utils.flatten(nodes_popped)

    states_search: List[State] = [node.state for node in nodes_popped_flat]

    per_reached_sg: float = 100 * np.mean(astar.has_found_goal())

    return states_search, per_reached_sg


def try_to_reach_subgoal_gbfs(states: List[State], eps_max: float, env: Environment, heuristic_fn, num_steps: int):
    eps: List[float] = list(np.random.rand(len(states)) * eps_max)

    gbfs = GBFS(states, env, eps=eps)
    for _ in range(num_steps):
        gbfs.step(heuristic_fn)

    trajs: List[List[Tuple[State, int, float]]] = gbfs.get_trajs()

    trajs_flat: List[Tuple[State, int, float]]
    trajs_flat, _ = misc_utils.flatten(trajs)

    states_search: List = []
    for traj in trajs_flat:
        states_search.append(traj[0])

    per_reached_sg: np.ndarray = 100 * np.mean(gbfs.get_is_solved())

    return states_search, per_reached_sg


def main():
    # arguments
    parser: ArgumentParser = ArgumentParser()
    args_dict: Dict[str, Any] = parse_arguments(parser)
    writer = SummaryWriter(log_dir=args_dict["model_dir"])

    if not args_dict["debug"]:
        sys.stdout = data_utils.Logger(args_dict["output_save_loc"], "a")

    # environment
    env: Environment = env_utils.get_environment(args_dict['env'], -1)
    print("Num actions: %i" % env.num_actions_max)

    # get device
    on_gpu: bool
    device: torch.device
    device, devices, on_gpu = nnet_utils.get_device()

    print("device: %s, devices: %s, on_gpu: %s" % (device, devices, on_gpu))

    itr_tot: int = get_itr_tot(args_dict["model_dir"])

    # get subgoal number
    subgoal_num: int = get_subgoal_num(args_dict["model_dir"], args_dict["subgoal_start"])

    # training
    envs_sg: List[Environment] = []
    for subgoal_num_prev in range(args_dict["subgoal_start"], subgoal_num):
        env_sg: Environment = env_utils.get_environment(args_dict['env'], subgoal_num_prev)
        envs_sg.append(env_sg)

    while subgoal_num <= env.num_subgoals:
        env_sg: Environment = env_utils.get_environment(args_dict['env'], subgoal_num)
        envs_sg.append(env_sg)

        target_dir = "%s/%s/target/" % (args_dict['model_dir'], subgoal_num)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        current_dir = "%s/%s/current/" % (args_dict['model_dir'], subgoal_num)
        if not os.path.exists(current_dir):
            os.makedirs(current_dir)

        # load nnet
        nnet, itr, update_num = load_nnet(current_dir, env_sg)

        nnet.to(device)
        if on_gpu:
            nnet = nn.DataParallel(nnet)

        per_reached_sg_astar: float = 0.0
        while per_reached_sg_astar < 100.0:
            print("-----Subgoal %i Update Num: %i-----" % (subgoal_num, update_num))
            writer.add_scalar('subgoal', subgoal_num, itr_tot)

            if update_num < 2:
                eps: float = 1.0
            else:
                eps: float = args_dict['eps']

            # generate states
            print("Generating states")
            start_time = time.time()
            num_samp_subgoal: int = args_dict['states_per_update'] * (1.0 - args_dict['frac_search'])
            num_samp_subgoal = int(np.ceil(num_samp_subgoal))

            num_samp_search: int = (args_dict['states_per_update'] - num_samp_subgoal) / args_dict[
                'update_search_steps']
            num_samp_search = int(np.ceil(num_samp_search))

            states_from_sg, _ = env_sg.generate_states(num_samp_subgoal, (0, args_dict['back_max']))

            states_scramb: List[State] = []
            if num_samp_search > 0:
                states_scramb, _ = env_sg.generate_states(num_samp_search, (100, 200))  # TODO make hyperparameter

            print("Num states - from subgoal: %i, scramb: %i" % (len(states_from_sg), len(states_scramb)))
            print("Time: %f\n" % (time.time() - start_time))

            # reach previous subgoal
            states_sg_prev: List[State] = states_scramb
            if (len(states_scramb) > 0) and (subgoal_num > args_dict["subgoal_start"]):
                print("Going from scrambled to previous subgoal")
                states_sg_prev = scrambled_to_subgoal(args_dict["subgoal_start"], subgoal_num - 1, states_scramb,
                                                      args_dict["astar_weight"], envs_sg, args_dict["model_dir"],
                                                      device, on_gpu, args_dict["update_nnet_batch_size"])
                print("")

            # update heuristic fn
            targ_file: str = "%s/model_state_dict.pt" % target_dir
            all_zeros: bool = not os.path.isfile(targ_file)
            if all_zeros:
                def heuristic_fn(x, *_):
                    return np.zeros((len(x), env_sg.num_actions_max), dtype=np.float)
            else:
                heuristic_fn = nnet_utils.load_heuristic_fn(target_dir, device, on_gpu, env_sg.get_nnet(), env_sg,
                                                            clip_zero=True,
                                                            batch_size=args_dict['update_nnet_batch_size'])

            # try to reach current subgoal
            if len(states_sg_prev) > 0:
                print("Trying to reach current subgoal from previous subgoal")
                start_time = time.time()
                states_to_sg, per_reached_sg = try_to_reach_subgoal_gbfs(states_sg_prev, eps, env_sg, heuristic_fn,
                                                                         args_dict["update_search_steps"])

                print("Percent reached subgoal: %f" % per_reached_sg)
                print("Number of states generated: %i" % len(states_to_sg))

                states_train = states_to_sg + states_from_sg

                print("Time: %f\n" % (time.time() - start_time))
            else:
                states_train = states_from_sg

            # update step with q-learning
            print("Updating with Q-learning step")
            start_time = time.time()

            ctgs, actions, _ = search_utils.q_step(states_train, heuristic_fn, env_sg, [eps] * len(states_train))
            print("Cost-to-go subgoal (mean/min/max): "
                  "%.2f/%.2f/%.2f" % (ctgs.mean(), ctgs.min(), ctgs.max()))

            states_train_nnet = env_sg.state_to_nnet_input(states_train)
            ctgs = np.expand_dims(ctgs, 1)
            print("Time: %f\n" % (time.time() - start_time))

            # train nnet
            num_train_itrs: int = args_dict['epochs_per_update'] * np.ceil(ctgs.shape[0] / args_dict['batch_size'])
            print("Training model for on %s examples for update number %i for %i "
                  "iterations" % (format(states_train_nnet[0].shape[0], ","), update_num, num_train_itrs))
            start_time = time.time()
            last_loss = nnet_utils.train_nnet(nnet, states_train_nnet, ctgs, actions, device,
                                              args_dict['batch_size'], num_train_itrs, itr, args_dict['lr'],
                                              args_dict['lr_d'])
            itr += num_train_itrs
            itr_tot += num_train_itrs
            print("Time: %f\n" % (time.time() - start_time))

            # save nnet
            pickle.dump(itr_tot, open("%s/itr_tot.pkl" % args_dict["model_dir"], "wb"), protocol=-1)

            torch.save(nnet.state_dict(), "%s/model_state_dict.pt" % current_dir)
            pickle.dump(itr, open("%s/train_itr.pkl" % current_dir, "wb"), protocol=-1)
            pickle.dump(update_num, open("%s/update_num.pkl" % current_dir, "wb"), protocol=-1)

            heuristic_fn = nnet_utils.get_heuristic_fn(nnet, device, env_sg,
                                                       batch_size=args_dict['update_nnet_batch_size'])

            # test with GBFS
            print("Testing with GBFS")
            start_time = time.time()
            max_solve_steps: int = min(update_num + 1, args_dict['back_max'])
            per_reached_sg_gbfs: float = gbfs_test(args_dict['num_test'], args_dict['back_max'], env_sg, heuristic_fn,
                                                   max_solve_steps=max_solve_steps)
            writer.add_scalar('gbfs', per_reached_sg_gbfs, itr_tot)
            writer.add_scalar('gbfs_sg%i' % subgoal_num, per_reached_sg_gbfs, itr)

            writer.flush()
            print("Time: %f\n" % (time.time() - start_time))

            # test with A* search
            num_astar_steps: int = args_dict["astar_steps_incr"] * (update_num + 1)
            num_astar_steps = min(num_astar_steps, args_dict["max_astar_steps"])
            print("Testing with A* Search with %i steps" % num_astar_steps)

            start_time = time.time()
            states_scramb, _ = env_sg.generate_states(args_dict['num_test'], (100, 200))  # TODO make hyperparameter
            states_sg_prev = scrambled_to_subgoal(args_dict["subgoal_start"], subgoal_num - 1, states_scramb,
                                                  args_dict["astar_weight"], envs_sg, args_dict["model_dir"], device,
                                                  on_gpu, args_dict["update_nnet_batch_size"])

            astar = AStar(states_sg_prev, env_sg, heuristic_fn,
                          weights=[args_dict["astar_weight"]] * len(states_sg_prev))

            for _ in range(num_astar_steps):
                if not min(astar.has_found_goal()):
                    astar.step(heuristic_fn, 1, verbose=False)
            per_reached_sg_astar: float = 100 * np.mean(astar.has_found_goal())

            print("Percent reached subgoal: %f" % per_reached_sg_astar)
            writer.add_scalar('astar', per_reached_sg_astar, itr_tot)
            writer.add_scalar('astar_sg%i' % subgoal_num, per_reached_sg_astar, itr)
            print("Time: %f\n" % (time.time() - start_time))

            # clear cuda memory
            torch.cuda.empty_cache()

            # print("Last loss was %f" % last_loss)
            if last_loss < args_dict['loss_thresh'] or True:
                # Update nnet
                print("Updating target network")
                copy_files(current_dir, target_dir)
                update_num = update_num + 1
                pickle.dump(update_num, open("%s/update_num.pkl" % current_dir, "wb"), protocol=-1)

            print("")

        subgoal_num += 1
        pickle.dump(subgoal_num, open("%s/subgoal_num.pkl" % args_dict["model_dir"], "wb"), protocol=-1)

    writer.close()

    print("Done")


if __name__ == "__main__":
    main()
