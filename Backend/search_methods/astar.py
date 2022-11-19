from typing import List, Tuple, Dict, Callable, Optional, Any, Set
from environments.environment_abstract import Environment, State, MacroAction
import numpy as np
from heapq import heappush, heappop
from subprocess import Popen, PIPE, STDOUT

from argparse import ArgumentParser
import torch
from utils import env_utils, nnet_utils, search_utils, misc_utils, data_utils
import pickle
import time
import sys
import os


class Node:
    __slots__ = ['state', 'path_cost', 'heuristic', 'cost', 'is_solved', 'parent_move', 'parent', 'transition_costs',
                 'children', 'bellman']

    def __init__(self, state: State, path_cost: float, is_solved: bool,
                 parent_move: Optional[int], parent):
        self.state: State = state
        self.path_cost: float = path_cost
        self.heuristic: Optional[float] = None
        self.cost: Optional[float] = None
        self.is_solved: bool = is_solved
        self.parent_move: Optional[int] = parent_move
        self.parent: Optional[Node] = parent

        self.transition_costs: List[float] = []
        self.children: List[Node] = []

        self.bellman: float = np.inf

    def compute_bellman(self):
        if self.is_solved:
            self.bellman = 0.0
        elif len(self.children) == 0:
            self.bellman = self.heuristic
        else:
            for node_c, tc in zip(self.children, self.transition_costs):
                self.bellman = min(self.bellman, tc + node_c.heuristic)


OpenSetElem = Tuple[float, int, Node]


class Instance:

    def __init__(self, root_node: Node, state_goal: State, env: Environment,
                 macro_actions_forbid: Optional[Set[MacroAction]]):
        self.state_goal: State = state_goal

        self.open_set: List[OpenSetElem] = []
        self.heappush_count: int = 0
        self.closed_dict: Dict[State, float] = dict()
        self.popped_nodes: List[Node] = []
        self.goal_nodes: List[Node] = []
        self.num_nodes_generated: int = 0

        self.root_node: Node = root_node
        self.env = env
        self.macro_actions_forbid: Optional[Set[MacroAction]] = macro_actions_forbid

        self.push_to_open([self.root_node])

    def push_to_open(self, nodes: List[Node]):
        for node in nodes:
            heappush(self.open_set, (node.cost, self.heappush_count, node))
            self.heappush_count += 1

    def pop_from_open(self, num_nodes: int) -> List[Node]:
        num_to_pop: int = min(num_nodes, len(self.open_set))

        popped_nodes = [heappop(self.open_set)[2] for _ in range(num_to_pop)]
        for node in popped_nodes:
            if node.is_solved:
                if self.macro_actions_forbid is not None:
                    _, action_seq, _ = get_path(node)
                    macro_action: MacroAction = self.env.get_macro_action(action_seq)
                    if macro_action not in self.macro_actions_forbid:
                        self.goal_nodes.append(node)
                    else:
                        self.closed_dict.pop(node.state)
                else:
                    self.goal_nodes.append(node)
        # self.goal_nodes.extend([node for node in popped_nodes if node.is_solved])
        self.popped_nodes.extend(popped_nodes)

        return popped_nodes

    def remove_in_closed(self, nodes: List[Node]) -> List[Node]:
        nodes_not_in_closed: List[Node] = []

        for node in nodes:
            path_cost_prev: Optional[float] = self.closed_dict.get(node.state)
            if path_cost_prev is None:
                nodes_not_in_closed.append(node)
                self.closed_dict[node.state] = node.path_cost
            elif path_cost_prev > node.path_cost:
                nodes_not_in_closed.append(node)
                self.closed_dict[node.state] = node.path_cost

        return nodes_not_in_closed


def pop_from_open(instances: List[Instance], batch_size: int) -> List[List[Node]]:
    popped_nodes_all: List[List[Node]] = [instance.pop_from_open(batch_size) for instance in instances]

    return popped_nodes_all


def expand_nodes(instances: List[Instance], popped_nodes_all: List[List[Node]], env: Environment):
    # Get children of all nodes at once (for speed)
    states_goal_all: List[List[State]] = []
    for instance, popped_nodes_inst in zip(instances, popped_nodes_all):
        states_goal_all.append([instance.state_goal] * len(popped_nodes_inst))

    popped_nodes_flat: List[Node]
    split_idxs: List[int]
    popped_nodes_flat, split_idxs = misc_utils.flatten(popped_nodes_all)

    states_goal_flat, _ = misc_utils.flatten(states_goal_all)

    if len(popped_nodes_flat) == 0:
        return [[]]

    states_flat: List[State] = [x.state for x in popped_nodes_flat]

    states_c_by_node: List[List[State]]
    tcs_np: List[np.ndarray]

    states_c_by_node, tcs_np = env.expand(states_flat)
    states_c_goal_by_node: List[List[State]] = []
    for state_goal, states_c_node in zip(states_goal_flat, states_c_by_node):
        states_c_goal_by_node.append([state_goal] * len(states_c_node))

    tcs_by_node: List[List[float]] = [list(x) for x in tcs_np]

    # Get is_solved on all states at once (for speed)
    states_c: List[State]

    states_c, split_idxs_c = misc_utils.flatten(states_c_by_node)
    states_c_goal, _ = misc_utils.flatten(states_c_goal_by_node)

    is_solved_c: List[bool] = list(env.is_solved(states_c, states_c_goal))
    is_solved_c_by_node: List[List[bool]] = misc_utils.unflatten(is_solved_c, split_idxs_c)

    # Update path costs for all states at once (for speed)
    parent_path_costs = np.expand_dims(np.array([node.path_cost for node in popped_nodes_flat]), 1)
    path_costs_c: List[float] = (parent_path_costs + np.array(tcs_by_node)).flatten().tolist()

    path_costs_c_by_node: List[List[float]] = misc_utils.unflatten(path_costs_c, split_idxs_c)

    # Reshape lists
    tcs_by_inst_node: List[List[List[float]]] = misc_utils.unflatten(tcs_by_node, split_idxs)
    patch_costs_c_by_inst_node: List[List[List[float]]] = misc_utils.unflatten(path_costs_c_by_node,
                                                                               split_idxs)
    states_c_by_inst_node: List[List[List[State]]] = misc_utils.unflatten(states_c_by_node, split_idxs)
    is_solved_c_by_inst_node: List[List[List[bool]]] = misc_utils.unflatten(is_solved_c_by_node, split_idxs)

    # Get child nodes
    instance: Instance
    nodes_c_by_inst: List[List[Node]] = []
    states_goal_c_by_inst: List[List[State]] = []
    for inst_idx, instance in enumerate(instances):
        nodes_c_by_inst.append([])
        states_goal_c_by_inst.append([])

        parent_nodes: List[Node] = popped_nodes_all[inst_idx]
        tcs_by_node: List[List[float]] = tcs_by_inst_node[inst_idx]
        path_costs_c_by_node: List[List[float]] = patch_costs_c_by_inst_node[inst_idx]
        states_c_by_node: List[List[State]] = states_c_by_inst_node[inst_idx]

        is_solved_c_by_node: List[List[bool]] = is_solved_c_by_inst_node[inst_idx]

        parent_node: Node
        tcs_node: List[float]
        states_c: List[State]
        str_reps_c: List[str]
        for parent_node, tcs_node, path_costs_c, states_c, is_solved_c in zip(parent_nodes, tcs_by_node,
                                                                              path_costs_c_by_node, states_c_by_node,
                                                                              is_solved_c_by_node):
            state: State
            for move_idx, state in enumerate(states_c):
                path_cost: float = path_costs_c[move_idx]
                is_solved: bool = is_solved_c[move_idx]
                node_c: Node = Node(state, path_cost, is_solved, move_idx, parent_node)

                nodes_c_by_inst[inst_idx].append(node_c)
                states_goal_c_by_inst[inst_idx].append(instance.state_goal)

                parent_node.children.append(node_c)

            parent_node.transition_costs.extend(tcs_node)

        instance.num_nodes_generated += len(nodes_c_by_inst[inst_idx])

    return nodes_c_by_inst, states_goal_c_by_inst


def remove_in_closed(instances: List[Instance], nodes_c_all: List[List[Node]]) -> List[List[Node]]:
    for inst_idx, instance in enumerate(instances):
        nodes_c_all[inst_idx] = instance.remove_in_closed(nodes_c_all[inst_idx])

    return nodes_c_all


def add_heuristic_and_cost(nodes: List[Node], states_goal: List[State], heuristic_fn: Callable,
                           weights: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    # flatten nodes
    nodes: List[Node]

    if len(nodes) == 0:
        return np.zeros(0), np.zeros(0)

    # get heuristic
    states: List[State] = [node.state for node in nodes]

    # compute node cost
    heuristics = heuristic_fn(states, states_goal)
    path_costs: np.ndarray = np.array([node.path_cost for node in nodes])
    is_solved: np.ndarray = np.array([node.is_solved for node in nodes])

    costs: np.ndarray = np.array(weights) * path_costs + heuristics * np.logical_not(is_solved)

    # add cost to node
    for node, heuristic, cost in zip(nodes, heuristics, costs):
        node.heuristic = heuristic
        node.cost = cost

    return path_costs, heuristics


def add_to_open(instances: List[Instance], nodes: List[List[Node]]) -> None:
    nodes_inst: List[Node]
    instance: Instance
    for instance, nodes_inst in zip(instances, nodes):
        instance.push_to_open(nodes_inst)


def get_path(node: Node) -> Tuple[List[State], List[int], float]:
    path: List[State] = []
    moves: List[int] = []

    parent_node: Node = node
    while parent_node.parent is not None:
        path.append(parent_node.state)

        moves.append(parent_node.parent_move)
        parent_node = parent_node.parent

    path.append(parent_node.state)

    path = path[::-1]
    moves = moves[::-1]

    return path, moves, node.path_cost


class AStar:

    def __init__(self, states: List[State], states_goal: List[State], env: Environment, heuristic_fn: Callable,
                 weights: List[float], macro_actions_forbid: Optional[Set[MacroAction]] = None):
        self.env: Environment = env
        self.weights: List[float] = weights
        self.step_num: int = 0

        self.timings: Dict[str, float] = {"pop": 0.0, "expand": 0.0, "check": 0.0, "heur": 0.0,
                                          "add": 0.0, "itr": 0.0}

        # compute starting costs
        root_nodes: List[Node] = []
        is_solved_states: np.ndarray = self.env.is_solved(states, states_goal)
        for state, is_solved in zip(states, is_solved_states):
            root_node: Node = Node(state, 0.0, is_solved, None, None)
            root_nodes.append(root_node)

        add_heuristic_and_cost(root_nodes, states_goal, heuristic_fn, self.weights)

        # initialize instances
        self.instances: List[Instance] = []
        for root_node, state_goal in zip(root_nodes, states_goal):
            self.instances.append(Instance(root_node, state_goal, env, macro_actions_forbid))

    def step(self, heuristic_fn: Callable, batch_size: int, include_solved: bool = False, verbose: bool = False):
        start_time_itr = time.time()
        instances: List[Instance]
        if include_solved:
            instances = self.instances
        else:
            instances = [instance for instance in self.instances if len(instance.goal_nodes) == 0]

        # Pop from open
        start_time = time.time()
        popped_nodes_all: List[List[Node]] = pop_from_open(instances, batch_size)
        pop_time = time.time() - start_time

        # Expand nodes
        start_time = time.time()
        nodes_c_all: List[List[Node]]
        states_goal_c_all: List[List[State]]
        nodes_c_all, states_goal_c_all = expand_nodes(instances, popped_nodes_all, self.env)
        expand_time = time.time() - start_time

        # Get heuristic of children, do heur before check so we can do backup
        start_time = time.time()
        nodes_c_all_flat, _ = misc_utils.flatten(nodes_c_all)
        states_goal_c_all_flat, _ = misc_utils.flatten(states_goal_c_all)
        weights, _ = misc_utils.flatten([[weight] * len(nodes_c) for weight, nodes_c in zip(self.weights, nodes_c_all)])
        path_costs, heuristics = add_heuristic_and_cost(nodes_c_all_flat, states_goal_c_all_flat, heuristic_fn, weights)
        heur_time = time.time() - start_time

        # Check if children are in closed
        start_time = time.time()
        nodes_c_all = remove_in_closed(instances, nodes_c_all)
        check_time = time.time() - start_time

        # Add to open
        start_time = time.time()
        add_to_open(instances, nodes_c_all)
        add_time = time.time() - start_time

        itr_time = time.time() - start_time_itr

        # Print to screen
        if verbose:
            if heuristics.shape[0] > 0:
                min_heur = np.min(heuristics)
                min_heur_pc = path_costs[np.argmin(heuristics)]
                max_heur = np.max(heuristics)
                max_heur_pc = path_costs[np.argmax(heuristics)]

                print("Itr: %i, Added to OPEN - Min/Max Heur(PathCost): "
                      "%.2f(%.2f)/%.2f(%.2f) " % (self.step_num, min_heur, min_heur_pc, max_heur, max_heur_pc))

            per_solved: float = 100 * float(np.mean(self.has_found_goal()))
            print("%% Solved: %.2f, Times - pop: %.2f, expand: %.2f, check: %.2f, heur: %.2f, "
                  "add: %.2f, itr: %.2f" % (per_solved, pop_time, expand_time, check_time, heur_time, add_time,
                                            itr_time))

            print("")

        # Update timings
        self.timings['pop'] += pop_time
        self.timings['expand'] += expand_time
        self.timings['check'] += check_time
        self.timings['heur'] += heur_time
        self.timings['add'] += add_time
        self.timings['itr'] += itr_time

        self.step_num += 1

    def has_found_goal(self) -> List[bool]:
        goal_found: List[bool] = [len(self.get_goal_nodes(idx)) > 0 for idx in range(len(self.instances))]

        return goal_found

    def get_goal_nodes(self, inst_idx) -> List[Node]:
        return self.instances[inst_idx].goal_nodes

    def get_cheapest_goal_node(self, inst_idx) -> Node:
        goal_nodes: List[Node] = self.get_goal_nodes(inst_idx)
        path_costs: List[float] = [node.path_cost for node in goal_nodes]

        goal_node: Node = goal_nodes[int(np.argmin(path_costs))]

        return goal_node

    def get_num_nodes_generated(self, inst_idx: int) -> int:
        return self.instances[inst_idx].num_nodes_generated

    def get_popped_nodes(self) -> List[List[Node]]:
        popped_nodes_all: List[List[Node]] = [instance.popped_nodes for instance in self.instances]
        return popped_nodes_all


def main():
    # parse arguments
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--states', type=str, required=True, help="File containing states to solve")
    parser.add_argument('--heur', type=str, required=True, help="Directory of nnet model")
    parser.add_argument('--env', type=str, required=True, help="Environment: cube3, 15-puzzle, 24-puzzle")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for BWAS")
    parser.add_argument('--weight', type=float, default=1.0, help="Weight of path cost")
    parser.add_argument('--subgoal_num', type=int, default=0, help="")
    parser.add_argument('--language', type=str, default="python", help="python or cpp")

    parser.add_argument('--results_dir', type=str, required=True, help="Directory to save results")
    parser.add_argument('--start_idx', type=int, default=0, help="")
    parser.add_argument('--nnet_batch_size', type=int, default=None, help="Set to control how many states per GPU are "
                                                                          "evaluated by the neural network at a time. "
                                                                          "Does not affect final results, "
                                                                          "but will help if nnet is running out of "
                                                                          "memory.")

    parser.add_argument('--redo', action='store_true', default=False, help="Set to start from scratch")
    parser.add_argument('--verbose', action='store_true', default=False, help="Set for verbose")
    parser.add_argument('--debug', action='store_true', default=False, help="Set when debugging")

    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    results_file: str = "%s/results.pkl" % args.results_dir
    output_file: str = "%s/output.txt" % args.results_dir
    if not args.debug:
        sys.stdout = data_utils.Logger(output_file, "w")

    # get data
    input_data = pickle.load(open(args.states, "rb"))
    states: List[State] = input_data['states'][args.start_idx:]
    states_goal: List[State] = input_data['state_goals'][args.start_idx:]

    # environment
    env: Environment = env_utils.get_environment(args.env, args.subgoal_num)

    # initialize results
    has_results: bool = False
    if os.path.isfile(results_file):
        has_results = True

    if has_results and (not args.redo):
        results: Dict[str, Any] = pickle.load(open(results_file, "rb"))
        if not args.debug:
            sys.stdout = data_utils.Logger(output_file, "a")
    else:
        results: Dict[str, Any] = {"states": states, "solutions": [], "paths": [], "iterations": [], "times": [],
                                   "num_nodes_generated": []}
        if not args.debug:
            sys.stdout = data_utils.Logger(output_file, "w")

    if args.language == "python":
        bwas_python(args, env, states, states_goal)
    elif args.language == "cpp":
        bwas_cpp(args, env, states, states_goal, results, results_file)
    else:
        raise ValueError("Unknown language %s" % args.language)

    pickle.dump(results, open(results_file, "wb"), protocol=-1)


def bwas_python(args, env: Environment, states: List[State], states_goal: List[State]):
    # get device
    on_gpu: bool
    device: torch.device
    device, devices, on_gpu = nnet_utils.get_device()

    print("device: %s, devices: %s, on_gpu: %s" % (device, devices, on_gpu))

    heuristic_fn = nnet_utils.load_heuristic_fn(args.heur, device, on_gpu, env.get_nnet(),
                                                env, clip_zero=True, batch_size=args.nnet_batch_size)

    solns: List[List[int]] = []
    paths: List[List[State]] = []
    times: List = []
    num_nodes_gen: List[int] = []

    for state_idx, state in enumerate(states):
        start_time = time.time()

        state_goal: State = states_goal[state_idx]

        num_itrs: int = 0
        astar = AStar([state], [state_goal], env, heuristic_fn, [args.weight])
        while not min(astar.has_found_goal()):
            astar.step(heuristic_fn, args.batch_size, verbose=args.verbose)
            num_itrs += 1

        path: List[State]
        soln: List[int]
        path_cost: float
        num_nodes_gen_idx: int
        goal_node: Node = astar.get_cheapest_goal_node(0)
        path, soln, path_cost = get_path(goal_node)

        num_nodes_gen_idx: int = astar.get_num_nodes_generated(0)

        solve_time = time.time() - start_time

        # record solution information
        solns.append(soln)
        paths.append(path)
        times.append(solve_time)
        num_nodes_gen.append(num_nodes_gen_idx)

        # check soln
        assert search_utils.is_valid_soln(state, state_goal, soln, env)

        # print to screen
        timing_str = ", ".join(["%s: %.2f" % (key, val) for key, val in astar.timings.items()])
        print("Times - %s, num_itrs: %i" % (timing_str, num_itrs))

        print("State: %i, SolnCost: %.2f, # Moves: %i, "
              "# Nodes Gen: %s, Time: %.2f" % (state_idx, path_cost, len(soln),
                                               format(num_nodes_gen_idx, ","),
                                               solve_time))

    return solns, paths, times, num_nodes_gen


def bwas_cpp(args, env: Environment, states: List[State], states_goal: List[State], results: Dict[str, Any],
             results_file: str):
    assert (args.env.upper() in ['CUBE3', 'CUBE3_2', 'CUBE3_3', 'CUBE4', 'PUZZLE15', 'PUZZLE24', 'PUZZLE35',
                                 'PUZZLE48']) or ('LIGHTSOUT' in args.env.upper())

    # start heuristic proc
    traced_model = "%s/model_traced.pt" % args.heur

    # Make c++ input file
    cpp_input_file: str = "%s_cpp_input.txt" % results_file.split(".pkl")[0]

    state_idx: int = len(results["solutions"])

    with open(cpp_input_file, 'w') as file:
        state_idx_write: int = state_idx
        while state_idx_write < len(states):
            state = states[state_idx_write]
            state_goal = states_goal[state_idx_write]
            # Get string rep of state
            if args.env.upper() in ['CUBE3', 'CUBE3_2', 'CUBE3_3']:
                state_cpp: np.array = env.state_to_nnet_input([state])[0][0]
                state_goal_cpp: np.array = env.state_to_nnet_input([state_goal])[0][0]
            elif args.env.upper() in ["PUZZLE15", "PUZZLE24", "PUZZLE35", "PUZZLE48"]:
                state_cpp: np.array = state.tiles
                state_goal_cpp: np.array = state_goal.tiles
            else:
                raise ValueError("Unknown c++ environment: %s" % args.env)

            states_cpp_str = " ".join([str(x) for x in state_cpp])
            file.write(states_cpp_str + "\n")

            states_goal_cpp_str = " ".join([str(x) for x in state_goal_cpp])
            file.write(states_goal_cpp_str + "\n")

            state_idx_write += 1

    # run c++
    popen = Popen(['./cpp/build/parallel_weighted_astar', str(args.weight), str(args.batch_size),
                   cpp_input_file, args.env, traced_model, str(args.nnet_batch_size)], stdout=PIPE, stderr=STDOUT,
                  bufsize=1, universal_newlines=True)

    # get results
    lines = []
    for stdout_line in iter(popen.stdout.readline, ""):
        stdout_line = stdout_line.strip('\n')
        lines.append(stdout_line)
        if args.verbose:
            sys.stdout.write("%s\n" % stdout_line)
            sys.stdout.flush()

        if (len(lines) > 1) and (lines[-2] == "Total time:"):
            state: State = states[state_idx]
            state_goal: State = states_goal[state_idx]

            num_itrs = int(lines[-7])
            moves = [int(x) for x in lines[-5].split(" ")[:-1]]
            soln = [x for x in moves][::-1]
            num_nodes_gen_idx = int(lines[-3])
            solve_time = float(lines[-1])

            # record solution information
            path: List[State] = [state]
            next_state: State = state
            transition_costs: List[float] = []

            for move in soln:
                next_states, tcs = env.next_state([next_state], [move])

                next_state = next_states[0]
                tc = tcs[0]

                path.append(next_state)
                transition_costs.append(tc)

            results["solutions"].append(soln)
            results["paths"].append(path)
            results["iterations"].append(num_itrs)
            results["times"].append(solve_time)
            results["num_nodes_generated"].append(num_nodes_gen_idx)

            path_cost: float = sum(transition_costs)

            # check soln
            assert search_utils.is_valid_soln(state, state_goal, soln, env)

            # print to screen
            print("State: %i, SolnCost: %.2f, # Moves: %i, # Nodes Gen: %s, Itrs: %i, Itrs/sec: %.2f, "
                  "Time: %.2f" % (state_idx, path_cost, len(soln), format(num_nodes_gen_idx, ","), num_itrs,
                                  num_itrs/solve_time, solve_time))

            pickle.dump(results, open(results_file, "wb"), protocol=-1)

            state_idx += 1


if __name__ == "__main__":
    main()
