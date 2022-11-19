from typing import List, Dict, Tuple, Union, Any, Set, Optional
import numpy as np

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from random import randrange

from utils.pytorch_models import FullyConnectedModel, ResnetModel
from utils import bias_utils, misc_utils
from utils.bias_utils import BiasPredicate
from popper.core import Literal, Clause, Program
from .environment_abstract import Environment, State, MacroAction

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
from visualizers.cube3_viz_simple import InteractiveCube
import re
import random


class Cube3ProcessStates(nn.Module):
    def _forward_unimplemented(self, *input_val: Any) -> None:
        pass

    def __init__(self, state_dim: int, one_hot_depth: int):
        super().__init__()
        self.state_dim: int = state_dim
        self.one_hot_depth: int = one_hot_depth

    def forward(self, states_nnet: Tensor):
        x = states_nnet

        # preprocess input
        if self.one_hot_depth > 0:
            x = F.one_hot(x.long(), self.one_hot_depth)
            x = x.float()
            x = x.view(-1, self.state_dim * self.one_hot_depth)
        else:
            x = x.float()

        return x


class Cube3FCResnet(nn.Module):
    def _forward_unimplemented(self, *input_val: Any) -> None:
        pass

    def __init__(self, input_dim: int, h1_dim: int, resnet_dim: int, num_resnet_blocks: int, out_dim: int,
                 batch_norm: bool):
        super().__init__()
        self.first_fc = FullyConnectedModel(input_dim, [h1_dim, resnet_dim], [batch_norm] * 2, ["RELU"] * 2)
        self.resnet = ResnetModel(resnet_dim, num_resnet_blocks, out_dim, batch_norm)

    def forward(self, x: Tensor):
        x = self.first_fc(x)
        x = self.resnet(x)

        return x


class Cube3NNet(nn.Module):
    def _forward_unimplemented(self, *input_val: Any) -> None:
        pass

    def __init__(self, state_dim: int, one_hot_depth: int, h1_dim: int, resnet_dim: int, num_res_blocks: int,
                 out_dim: int, batch_norm: bool):
        super().__init__()
        self.state_proc = Cube3ProcessStates(state_dim, one_hot_depth)

        input_dim: int = state_dim * one_hot_depth * 2
        self.dqn = Cube3FCResnet(input_dim, h1_dim, resnet_dim, num_res_blocks, out_dim, batch_norm)

    def forward(self, states, states_goal):
        states_proc = self.state_proc(states)
        states_goal_proc = self.state_proc(states_goal)

        x = self.dqn(torch.cat((states_proc, states_goal_proc), dim=1))

        return x


class Cube3State(State):
    __slots__ = ['colors', 'hash']

    def __init__(self, colors: np.array):
        self.colors: np.array = colors
        self.hash = None

    def __hash__(self):
        if self.hash is None:
            self.hash = hash(self.colors.tostring())

        return self.hash

    def __eq__(self, other):
        return np.array_equal(self.colors, other.colors)


class Cube3MacroAction(MacroAction):
    def __init__(self, action_seq_syms: Set[Tuple[int, ...]], prog: Program, clause_orig: Clause):
        self.action_seq_syms: Set[Tuple[int, ...]] = action_seq_syms
        self.clause_orig: Clause = clause_orig
        self.complexity: float = float(len(next(iter(self.action_seq_syms))))
        self.prog = prog
        self.precond: Optional[Program] = None
        self.hash = None

    def get_macro_action(self) -> Program:
        return self.prog

    def set_precond(self, precond: Program):
        self.precond = precond

    def get_precond(self):
        return self.precond

    def get_complexity(self) -> float:
        return self.complexity

    def to_string(self) -> str:
        return self.clause_orig.to_code()

    def __hash__(self):
        if self.hash is None:
            self.hash = 0  # TODO make better hash

        return self.hash

    def __eq__(self, other):
        action_seq_sym_elem: Tuple[int, ...] = next(iter(self.action_seq_syms))
        return action_seq_sym_elem in other.action_seq_syms


def _convert_subgoal_np(states_np: np.ndarray, ignore: np.array):
    if ignore.shape[0] > 0:
        argsort_idxs = np.argsort(states_np, axis=1)
        first_idxs = np.stack([np.arange(0, states_np.shape[0])] * ignore.shape[0], axis=1)

        states_np[first_idxs, argsort_idxs[:, ignore]] = 54


class Cube3(Environment):
    atomic_actions: List[str] = ["%s%i" % (f, n) for f in ['U', 'D', 'L', 'R', 'B', 'F'] for n in [-1, 1]]

    def __init__(self, subgoal_num: int):
        super().__init__()
        self.dtype = np.uint8
        self.cube_len: int = 3
        self.faces: List[str] = ['w', 'y', 'o', 'r', 'b', 'g', 'z']

        # all actions
        # action_combs: List[List[int]] = [[x] for x in range(len(self.atomic_actions))]
        # for i in range(0, len(self.atomic_actions), 2):
        #    action_combs.append([i, i])

        # self.action_combs = action_combs
        self.num_actions = len(self.atomic_actions)

        # solved state
        self.goal_colors: np.ndarray = np.arange(0, (self.cube_len ** 2) * 6, 1, dtype=self.dtype)

        cbs_m = [[4], [13], [22], [31], [40], [49]]
        cbs_e = [[1, 23], [3, 50], [5, 41], [7, 32],
                 [10, 21], [12, 39], [14, 48], [16, 30],
                 [19, 43], [25, 46],
                 [28, 52], [34, 37]]
        cbs_c = [[0, 26, 47], [2, 20, 44], [6, 29, 53], [8, 35, 38],
                 [9, 18, 42], [11, 24, 45], [15, 33, 36], [17, 27, 51]]

        self.sticker_num_to_cubelet_num: Dict[int, int] = dict()
        self.cubelets = cbs_m + cbs_e + cbs_c
        for cubelet_num, cubelet in enumerate(self.cubelets):
            for sticker_num in cubelet:
                self.sticker_num_to_cubelet_num[sticker_num] = cubelet_num

        self.sticker_num_to_cubelet: Dict[int, str] = dict()
        self.cubelet_to_stickers: Dict[str, List[int]] = dict()
        self.cubelet_to_color_to_num: Dict[str, Dict[str, int]] = dict()
        for cubelet in self.cubelets:
            cubelet_name: str = self._cubelet_to_name(cubelet)
            self.cubelet_to_stickers[cubelet_name] = cubelet.copy()
            self.cubelet_to_color_to_num[cubelet_name] = dict()
            for sticker_num in cubelet:
                self.sticker_num_to_cubelet[sticker_num] = cubelet_name

                color_name: str = self.faces[sticker_num // (self.cube_len ** 2)]
                self.cubelet_to_color_to_num[cubelet_name][color_name] = sticker_num

        self.sticker_num_to_cubelet[54] = 'blank'

        subgoal0 = [cbs_m[0], cbs_m[5], cbs_e[1]]
        # subgoal1 = cbs_m[0] + cbs_e[0] + cbs_e[1] + cbs_e[2] + cbs_e[3] + cbs_c[0] + cbs_c[1] + cbs_c[2] + cbs_c[3]
        # subgoal1 = [cbs_c[0], cbs_c[1], cbs_c[2], cbs_c[3]]
        # subgoal1.extend(cbs_m)

        subgoal1 = [cbs_e[0], cbs_e[1], cbs_e[2], cbs_e[3]]
        subgoal1.extend(cbs_m)
        # subgoal1 = subgoal0 + cbs_c[0] + cbs_c[1] + cbs_c[2] + cbs_c[3]
        # subgoal2 = subgoal1 + cbs_e[8] + cbs_e[9] + cbs_e[10] + cbs_e[11]
        # subgoal3 = subgoal2 + cbs_m[1] + cbs_e[4] + cbs_e[5] + cbs_e[6] + cbs_e[7]
        # subgoal4 = subgoal3 + cbs_c[4] + cbs_c[5] + cbs_c[6] + cbs_c[7]

        self.subgoal_cubelets = [subgoal0, subgoal1]

        # subgoal
        self.cbs_e = cbs_e
        if subgoal_num == -1:
            self.subgoal = np.array(list(range(self.goal_colors.shape[0])))
        else:
            subgoal_l = []
            for cubelet in self.subgoal_cubelets[subgoal_num]:
                subgoal_l.extend(cubelet)
            self.subgoal = np.array(subgoal_l)
        self.subgoal_num = subgoal_num

        self.ignore = np.array([i for i in range(len(self.goal_colors)) if i not in self.subgoal])

        # get idxs changed for moves
        self.rotate_idxs_new: Dict[str, np.ndarray]
        self.rotate_idxs_old: Dict[str, np.ndarray]

        self.adj_faces: Dict[int, np.ndarray]
        self._get_adj()

        self.rotate_idxs_new, self.rotate_idxs_old = self._compute_rotation_idxs(self.cube_len, self.atomic_actions)

        self.test_rep3d_state()
        self.sym_maps: List[Dict[int, int]] = self.get_sym_maps()
        # self.get_num_symmetries()
        # self.get_macro_action([0, 1, 2])

    @property
    def num_actions_max(self) -> int:
        return self.num_actions

    @property
    def num_subgoals(self) -> int:
        return 8

    def rand_action(self, states: List[State]) -> List[int]:
        return list(np.random.randint(0, self.num_actions, size=len(states)))

    def next_state(self, states: List[Cube3State], actions_l: List[int]) -> Tuple[List[Cube3State], List[float]]:
        states_np = np.stack([x.colors for x in states], axis=0)

        states_next_np = np.zeros(states_np.shape, dtype=self.dtype)
        tcs_np: np.array = np.zeros(len(states))
        actions = np.array(actions_l)
        for action in np.unique(actions):
            action_idxs = actions == action
            states_np_act = states_np[actions == action]

            # states_next_np_act_tmp = states_np_act
            # for atomic_action in self.action_combs[action]:
            #    states_next_np_act_tmp, _ = self._move_np(states_next_np_act_tmp, atomic_action)
            # states_next_np_act = states_next_np_act_tmp

            states_next_np_act, _ = self._move_np(states_np_act, action)

            # TODO assuming same transition cost
            tcs_act: List[float] = [1.0 for _ in range(states_np.shape[0])]

            states_next_np[action_idxs] = states_next_np_act
            tcs_np[action_idxs] = np.array(tcs_act)

        states_next: List[Cube3State] = [Cube3State(x) for x in list(states_next_np)]
        transition_costs = list(tcs_np)

        return states_next, transition_costs

    def generate_goal_states(self, num_states: int, np_format: bool = False) -> Union[List[Cube3State], np.ndarray]:
        goal_np: np.ndarray = np.expand_dims(self.goal_colors.copy(), 0)
        solved_states_np: np.ndarray = np.repeat(goal_np, num_states, axis=0)
        _convert_subgoal_np(solved_states_np, self.ignore)

        if np_format:
            return solved_states_np
        else:
            solved_states: List[Cube3State] = [Cube3State(solved_state_np) for solved_state_np in solved_states_np]
            return solved_states

    def is_solved(self, states: List[Cube3State], states_goal: List[Cube3State]) -> np.array:
        states_np = np.stack([state.colors for state in states], axis=0)
        states_goal_np = np.stack([state.colors for state in states_goal], axis=0)

        is_equal = np.equal(states_np, states_goal_np)

        return np.all(is_equal, axis=1)

    def state_to_nnet_input(self, states: List[Cube3State]) -> List[np.ndarray]:
        states_np = np.stack([state.colors for state in states], axis=0)

        representation_np: np.ndarray = states_np / (self.cube_len ** 2)
        representation_np = representation_np.astype(self.dtype)
        # representation_np = np.eye(7)[representation_np].astype(self.dtype)
        # representation_np = representation_np.reshape((len(states), -1))

        representation: List[np.ndarray] = [representation_np]

        return representation

    def expand(self, states: List[Cube3State]) -> Tuple[List[List[Cube3State]], List[np.ndarray]]:
        # initialize
        num_states: int = len(states)

        states_exp: List[List[Cube3State]] = [[] for _ in range(len(states))]

        tc: np.ndarray = np.empty([num_states, self.num_actions_max])

        # for each move, get next states, transition costs, and if solved
        for move_idx in range(self.num_actions_max):
            # next state
            states_next, tc_move = self.next_state(states, [move_idx] * len(states))

            # transition cost
            tc[:, move_idx] = np.array(tc_move)

            for idx in range(len(states)):
                states_exp[idx].append(states_next[idx])

        # make lists
        tc_l: List[np.ndarray] = [tc[i] for i in range(num_states)]

        return states_exp, tc_l

    def get_nnet(self) -> nn.Module:
        state_dim: int = (self.cube_len ** 2) * 6
        nnet = Cube3NNet(state_dim, 7, 5000, 1000, 4, 1, True)

        return nnet

    def state_to_predicate(self, states: List[Cube3State], convert: bool = True) -> List[str]:
        state_predicates: List[str] = []

        states_np = np.stack([state.colors for state in states], axis=0)

        for state_np in states_np:
            edge_str = ",".join([self.sticker_num_to_cubelet[x] for x in state_np])
            sticker_l: List[str] = [self.faces[x // (self.cube_len ** 2)] for x in state_np]
            sticker_l = [f"{x}_s" for x in sticker_l]
            sticker_str = ",".join(sticker_l)

            state_pred = f"state({edge_str},{sticker_str})"
            state_predicates.append(state_pred)

        return state_predicates

    def predicate_to_state(self, predicates: List[str]) -> List[Cube3State]:
        states: List[Cube3State] = []
        num_stickers: int = self.cube_len ** 2 * 6

        for predicate in predicates:
            predicate = misc_utils.remove_all_whitespace(predicate)
            match = re.search("state\((\S+)\)", predicate)
            cubelets_colors = match.group(1).split(",")

            cubelets = cubelets_colors[:num_stickers]
            colors = cubelets_colors[num_stickers:]
            colors = [x[0] for x in colors]

            colors_np = np.zeros(num_stickers, dtype=self.dtype)
            for idx, (cubelet, color) in enumerate(zip(cubelets, colors)):
                if cubelet == "blank":
                    sticker_num: int = num_stickers
                else:
                    sticker_num: int = self.cubelet_to_color_to_num[cubelet][color]
                colors_np[idx] = sticker_num

            state = Cube3State(colors_np)
            states.append(state)

        return states

    def generate_states(self, num_states: int, backwards_range: Tuple[int, int]) -> Tuple[List[Cube3State], List[int]]:
        assert (num_states > 0)
        assert (backwards_range[0] >= 0)
        assert self.fixed_actions, "Environments without fixed actions must implement their own method"

        # Initialize
        scrambs: List[int] = list(range(backwards_range[0], backwards_range[1] + 1))
        num_atomic_moves: int = len(self.atomic_actions)

        # Get goal states
        states_np: np.ndarray = self.generate_goal_states(num_states, np_format=True)

        # Scrambles
        scramble_nums: np.array = np.random.choice(scrambs, num_states)
        num_back_moves: np.array = np.zeros(num_states)

        # Go backward from goal state
        moves_lt = num_back_moves < scramble_nums
        while np.any(moves_lt):
            idxs: np.ndarray = np.where(moves_lt)[0]
            subset_size: int = int(max(len(idxs) / num_atomic_moves, 1))
            idxs: np.ndarray = np.random.choice(idxs, subset_size)

            move: int = randrange(num_atomic_moves)
            states_np[idxs], _ = self._move_np(states_np[idxs], move)

            num_back_moves[idxs] = num_back_moves[idxs] + 1
            moves_lt[idxs] = num_back_moves[idxs] < scramble_nums[idxs]

        states: List[Cube3State] = [Cube3State(x) for x in list(states_np)]

        return states, scramble_nums.tolist()

    def generate_focused_start_goals(self, states: List[Cube3State]) -> Tuple[List[Cube3State], List[Cube3State]]:
        states_start_np: np.ndarray = np.stack([state.colors for state in states], axis=0)
        states_focused_np: np.ndarray = self.generate_goal_states(len(states), np_format=True)

        sticker_nums: List[int] = misc_utils.flatten(self.subgoal_cubelets[self.subgoal_num])[0]
        cubelet_names: List[str] = list(set([self.sticker_num_to_cubelet[x] for x in sticker_nums]))

        for state_start_np, state_focused_np in zip(states_start_np, states_focused_np):
            random.shuffle(cubelet_names)
            found_mismatch_prev: bool = False
            for cubelet_name in cubelet_names:
                stk_idxs: np.array = np.array(self.cubelet_to_stickers[cubelet_name])
                if not np.array_equal(state_start_np[stk_idxs], state_focused_np[stk_idxs]):
                    if found_mismatch_prev:
                        for stk_idx in stk_idxs:
                            state_start_np[state_start_np == stk_idx] = 54
                            state_focused_np[state_focused_np == stk_idx] = 54
                    found_mismatch_prev = True

        states_start: List[Cube3State] = [Cube3State(x) for x in states_start_np]
        states_focused: List[Cube3State] = [Cube3State(x) for x in states_focused_np]

        return states_start, states_focused

    def generate_bk(self, file_name: str):
        file = open(file_name, "w")

        stickers: List[str] = ['w', 'y', 'o', 'r', 'b', 'g']
        color_names: List[str] = ['white', 'yellow', 'orange', 'red', 'blue', 'green']
        # layers: List[str] = ['l1', 'l2', 'l3']
        action_names: List[Tuple[str, str]] = [(f, n) for f in ['w', 'y', 'o', 'r', 'b', 'g'] for n in ["cc", "cl"]]
        action_names = action_names + [(f, "op") for f in ['w', 'y', 'o', 'r', 'b', 'g']]

        clockwise: Dict[str, List[str]] = dict()
        clockwise["w"] = ["b", "r", "g", "o"]
        clockwise["o"] = ["y", "b", "w", "g"]
        clockwise["g"] = ["o", "w", "r", "y"]

        opposite: Dict[str, str] = {"w": "y", "o": "r", "g": "b"}

        file.write("%% stickers and cubelets\n")
        for sticker in stickers:
            file.write(f"color({sticker}).\n")

        for sticker in stickers:
            file.write(f"sticker({sticker}_s).\n")

        for sticker in stickers:
            file.write(f"stk_col({sticker}_s,{sticker}).\n")

        for sticker in stickers:
            file.write(f"face({sticker}_f).\n")

        for sticker in stickers:
            file.write(f"face_col({sticker}_f,{sticker}).\n")

        file.write("\n")
        for color_name, sticker in zip(color_names, stickers):
            file.write(f"{color_name}({sticker}).\n")
            file.write(f"{color_name}({sticker}_f).\n")
            file.write(f"{color_name}({sticker}_s).\n")

        file.write("\n")

        file.write("\n%% Directions\n")
        file.write("cube_dir(cl).\n")
        file.write("cube_dir(cc).\n")
        file.write("cube_dir(op).\n")
        file.write("\n")

        file.write("clockwise(cl).\n"
                   "counterclockwise(cc).\n"
                   "opposite(op).\n")

        file.write("cube_dir_opposite(cc, cl).\n")
        file.write("cube_dir_opposite(cl, cc).\n")
        file.write("cube_dir_opposite(op, op).\n")
        file.write("\n")

        for center in clockwise.keys():
            centers_cl: List[str] = clockwise[center]
            for idx, center_cl in enumerate(centers_cl):
                idx_next = (idx + 1) % len(centers_cl)
                file.write(f"face_adjacent_dir({center}_f, {center_cl}_f, {centers_cl[idx_next]}_f, cl).\n")

        for center_op in clockwise.keys():
            center: str = opposite[center_op]
            centers_cl: List[str] = clockwise[center_op]
            for idx, center_cl in enumerate(centers_cl):
                idx_next = (idx - 1) % len(centers_cl)
                file.write(f"face_adjacent_dir({center}_f, {center_cl}_f, {centers_cl[idx_next]}_f, cl).\n")
        file.write("\n")

        file.write("face_adjacent_dir(A, B, C, cc) :- face_adjacent_dir(A, C, B, cl).\n")
        file.write("face_adjacent_dir(A, B, D, op) :- face_adjacent_dir(A, B, C, cl), "
                   "face_adjacent_dir(A, C, D, cl).\n")
        file.write("face_adjacent_dir(A, D, B, op) :- face_adjacent_dir(A, B, C, cl), "
                   "face_adjacent_dir(A, C, D, cl).\n")

        file.write("\n")
        file.write("face_rel_cl(A, Cbl, Stk, F2) :- onface(A, Cbl, Stk, F1), onface(A, Cbl, Stk2, F_ref), "
                   "face_adjacent_dir(F_ref, F1, F2, cl), Stk \= Stk2.\n")
        file.write("face_rel_cc(A, Cbl, Stk, F2) :- onface(A, Cbl, Stk, F1), onface(A, Cbl, Stk2, F_ref), "
                   "face_adjacent_dir(F_ref, F1, F2, cc), Stk \= Stk2.\n")
        file.write("face_rel_90(A, Cbl, Stk, F2) :- face_rel_cl(A, Cbl, Stk, F2).\n")
        file.write("face_rel_90(A, Cbl, Stk, F2) :- face_rel_cc(A, Cbl, Stk, F2).\n")
        file.write("face_rel_180(A, Cbl, Stk, F2) :- onface(A, Cbl, Stk, F1), onface(A, Cbl, Stk2, F_ref), "
                   "face_adjacent_dir(F_ref, F1, F2, op), Stk \= Stk2.\n")

        file.write("\n% onface\n")

        for sticker_num in range(54):
            edge_str: str = ",".join(["_"] * sticker_num + ["E"] + ["_"] * (54 - sticker_num - 1))
            sticker_str: str = ",".join(["_"] * sticker_num + ["S"] + ["_"] * (54 - sticker_num - 1))

            face_num: int = int(np.floor(sticker_num / 9))
            face: str = stickers[face_num]
            # file.write(f"onface(state({state_str}), E, S, {face}) :- num_to_edge(A, E), num_to_sticker(A, S).\n")
            file.write(f"onface(state({edge_str},{sticker_str}), E, S, {face}_f).\n")

        file.write("\n% move\n")
        for action_idx in range(self.num_actions_max):
            state: Cube3State = Cube3State(np.arange(0, 54, 1))
            state_next: Cube3State = self.next_state([state], [action_idx])[0][0]
            face, direc = action_names[action_idx]

            edge_str = ",".join([f"E{self.sticker_num_to_cubelet_num[num]}" for num in state.colors])
            edge_next_str = ",".join([f"E{self.sticker_num_to_cubelet_num[num]}" for num in state_next.colors])

            sticker_str = ",".join([f"S{num}" for num in state.colors])
            sticker_next_str = ",".join([f"S{num}" for num in state_next.colors])

            state_str: str = f"state({edge_str},{sticker_str})"
            state_next_str: str = f"state({edge_next_str},{sticker_next_str})"

            file.write(f"move({state_str}, {face}, {direc}, {state_next_str}).\n")
        file.write("\n")

        # file.write("p0(A):- face(F2, F2_col), face(F3, F3_col), cubelet(Cbl, Stk1, Stk2, Stk1_col, Stk2_col),"
        #           "white(F2_col), white(Stk2_col), onface(A, Cbl, Stk2, F3), face_rel_90(F2, F3), "
        #           "match_one(A, Cbl).\n")

        # file.write("p0(A):- face(F1), face(F2), face(F3), cubelet(Cbl), has_stk(Cbl, Stk2), has_stk(Cbl, Stk3), "
        #           "stk_col(Stk3, Stk3_col), face_col(F2, F2_col), stk_col(Stk2, F2_col), white(F2_col), "
        #           "onface(A, Cbl, Stk2, F3), onface(A, Cbl, Stk3, F1), face_rel_90(F2, F3), face_col(F1, F1_col), "
        #           "match(F1_col, Stk3_col).\n")

        # file.write("p0(A):- face(F2), face(F3), cubelet(Cbl), has_stk(Cbl, Stk1), has_stk(Cbl, Stk2), "
        #            "face_rel_90(A, Cbl, Stk1, F3), white(F3), white(Stk1),"
        #            "onface(A, Cbl, Stk2, F2), match(Stk2, F2).\n")

        # file.write("p0(A):- cubelet(Cbl), has_stk(Cbl, Stk1), has_stk(Cbl, Stk2), "
        #           "match_90(A, Cbl, Stk1), white(Stk1), match_0(A, Cbl, Stk2).\n")

        # file.write("p0(A):- cubelet(Cbl), has_stk(Cbl, Stk), "
        #           "match_90_s(A, Cbl, Stk), white(Stk), match_0_c(A, Cbl).\n")
        # file.write("p3_help(A, Cbl, Stk1, Stk2):- cubelet(Cbl), has_stk(Cbl, Stk1), has_stk(Cbl, Stk2), "
        #           "match_0_flip(A, Cbl, Stk1), white(Stk1), match_90(A, Cbl, Stk2), Stk1 \= Stk2.\n")

        # file.write("p3(A, Cbl):- face(F), has_stk(Cbl, Stk), onface(A, Cbl, Stk, F), stk_col(Stk, Stk_col), "
        #           "face_col(F, F_col), white(F_col), Stk_col \= F_col, adj_match_cl_cc(A, Cbl, Stk), "
        #           "match_none(A, Cbl).\n")

        file.write("\n% cubelet stickers\n")
        edge_preds: List[str] = []
        corner_preds: List[str] = []
        for cbl in self.cubelets:
            if len(cbl) == 1:
                continue

            cbl_name = self._cubelet_to_name(cbl)
            cbl_stks: List[str] = [stickers[stk_num // (self.cube_len ** 2)] for stk_num in cbl]
            for cbl_stk in cbl_stks:
                file.write(f"has_stk({cbl_name}, {cbl_stk}_s).\n")

            if len(cbl) == 2:
                edge_preds.append(f"edge({cbl_name})")
            elif len(cbl) == 3:
                corner_preds.append(f"corner({cbl_name})")

        file.write("\n% cubelets\n")
        for cbl_pred in edge_preds + corner_preds:
            file.write(f"{cbl_pred}.\n")

        file.write("\n")

        # set equivalence
        file.write("is_subset([],_).\n"
                   "is_subset([H|T],Y):- member(H,Y), select(H,Y,Z), is_subset(T,Z).\n"
                   "set_equal(X,Y):- is_subset(X,Y), is_subset(Y,X).\n")

        file.write("\n")

        file.write(
            "cubelet(Cbl) :- edge(Cbl).\n"
            "cubelet(Cbl) :- corner(Cbl).\n"
            
            "match_stk_f(Stk, F) :- stk_col(Stk, Stk_col), face_col(F, F_col), Stk_col = F_col.\n"
            "not_match_stk_f(Stk, F) :- stk_col(Stk, Stk_col), face_col(F, F_col), Stk_col \= F_col.\n"
            
            "match_0(A, Cbl, Stk) :- has_stk(Cbl, Stk), onface(A, Cbl, Stk, F), match_stk_f(Stk, F).\n"
            
            "match_0_flip(A, Cbl, Stk) :- edge(Cbl), face(F), has_stk(Cbl, Stk), has_stk(Cbl, Stk2), Stk \= Stk2, "
            "onface(A, Cbl, Stk2, F), match_stk_f(Stk, F).\n"

            "match_cl(A, Cbl, Stk) :- has_stk(Cbl, Stk), face_rel_cl(A, Cbl, Stk, F), match_stk_f(Stk, F).\n"
            "match_cc(A, Cbl, Stk) :- has_stk(Cbl, Stk), face_rel_cc(A, Cbl, Stk, F), match_stk_f(Stk, F).\n"
            "match_90(A, Cbl, Stk) :- has_stk(Cbl, Stk), face_rel_90(A, Cbl, Stk, F), match_stk_f(Stk, F).\n"
            
            "match_180(A, Cbl, Stk) :- has_stk(Cbl, Stk), face_rel_180(A, Cbl, Stk, F), match_stk_f(Stk, F).\n"
            
            "match_180_flip(A, Cbl, Stk) :- edge(Cbl), has_stk(Cbl, Stk), has_stk(Cbl, Stk2), Stk \= Stk2, "
            "face_rel_180(A, Cbl, Stk2, F_op), match_stk_f(Stk, F_op).\n"

            "not_in_place(A, Cbl) :- onface(A, Cbl, Stk, F), not_match_stk_f(Stk, F).\n"
            
            # "match_none(S, Cbl) :- onface(S, Cbl, Stk1, F1), onface(S, Cbl, Stk2, F2), face_col(F1, F1_col), "
            # "face_col(F2, F2_col), stk_col(Stk1, Stk1_col), stk_col(Stk2, Stk2_col), Stk1_col \= F1_col, "
            # "Stk2_col \= F2_col, Stk1 \= Stk2.\n"
            
            # "match_one(S, Cbl) :- onface(S, Cbl, Stk1, F1), onface(S, Cbl, Stk2, F2), face_col(F1, Col1), "
            # "stk_col(Stk1, Col1), face_col(F2, F2_col), stk_col(Stk2, Stk2_col), Stk2_col \= F2_col.\n"

            # "in_place(S, Cbl) :- edge(Cbl), onface(S, Cbl, Stk1, F1), onface(S, Cbl, Stk2, F2), face_col(F1, Col1), "
            # "stk_col(Stk1, Col1), face_col(F2, Col2), stk_col(Stk2, Col2), Stk1 \= Stk2.\n"
            
            "in_place(S, Cbl) :- edge(Cbl), onface(S, Cbl, Stk1, F1), onface(S, Cbl, Stk2, F2), match_stk_f(Stk1, F1), "
            "match_stk_f(Stk2, F2), Stk1 \= Stk2.\n"
            
            "in_place(S, Cbl) :- corner(Cbl), onface(S, Cbl, Stk1, F1), onface(S, Cbl, Stk2, F2), "
            "onface(S, Cbl, Stk3, F3), match_stk_f(Stk1, F1), match_stk_f(Stk2, F2), match_stk_f(Stk3, F3), "
            "Stk1 \= Stk2, Stk1 \= Stk3, Stk2 \= Stk3.\n"
            
            "in_place_set(S, IP_Set):- findall(X, in_place(S, X), IP_List), list_to_set(IP_List, IP_Set).\n"

            "in_place_subset(S1, S2) :- in_place_set(S1, IP_Set1), in_place_set(S2, IP_Set2), "
            "is_subset(IP_Set1, IP_Set2).\n"

            "in_place_diff(S1, S2, D) :- in_place_set(S1, IP_Set1), in_place_set(S2, IP_Set2), "
            "length(IP_Set1, L1), length(IP_Set2, L2), D is L1 - L2.\n"

            # "in_place_same(S1, S2) :- in_place_set(S1, IP_S1), in_place_set(S2, IP_S2), set_equal(IP_S1, IP_S2).\n"

            # "adj(S, Cbl, Stk, FRef, F1, F2, Rel) :- onface(S, Cbl, _, FRef), onface(S, Cbl, Stk, F1), "
            # "face_adjacent_dir(FRef, F1, F2, Rel).\n"
            # "adj_match_cl_cc(S, Cbl, Stk) :- adj(S, Cbl, Stk, _, _, F, cl), face_col(F, Col), stk_col(Stk, Col).\n"
            # "adj_match_cl_cc(S, Cbl, Stk) :- adj(S, Cbl, Stk, _, _, F, cc), face_col(F, Col), stk_col(Stk, Col).\n"
            # "adj_match_op(S, Cbl, Stk) :- adj(S, Cbl, Stk, _, _, F, op), face_col(F, Col), stk_col(Stk, Col).\n"
            # "adj_cbl_rel(S, Cbl, FRef, Rel) :- adj(S, Cbl, _, FRef, _, _, Rel).\n"
        )

        """
        file.write("\n% layers\n")
        for layer in layers:
            file.write(f"layer({layer}).\n")

        file.write("first_layer(l1).\n"
                   "second_layer(l2).\n"
                   "third_layer(l3).\n")

        file.write("on_layer(State, Edge, l1) :- onface(State, Edge, _, w_f).\n"

                   "on_layer(State, Edge, l2) :- onface(State, Edge, _, o_f), onface(State, Edge, _, g_f).\n"
                   "on_layer(State, Edge, l2) :- onface(State, Edge, _, g_f), onface(State, Edge, _, r_f).\n"
                   "on_layer(State, Edge, l2) :- onface(State, Edge, _, r_f), onface(State, Edge, _, b_f).\n"
                   "on_layer(State, Edge, l2) :- onface(State, Edge, _, b_f), onface(State, Edge, _, o_f).\n"

                   "on_layer(State, Edge, l3) :- onface(State, Edge, _, y_f).\n")
        """

        file.write("\n% goal\n")

        solved_pred_l: List[str] = []
        for cbl in self.subgoal_cubelets[self.subgoal_num]:
            if len(cbl) == 1:  # TODO should not have to skip
                continue
            cbl_name = self._cubelet_to_name(cbl)
            solved_pred_l.append(f"in_place(State, {cbl_name})")

        file.write(
            "human_heur(State, Count):- aggregate(sum(E), in_place(State, E), Count).\n"
            # "solved(State):- in_place_set(State, IP_Set), length(IP_Set, 4).\n"
            "solved(State):- %s.\n" % (", ".join(solved_pred_l))
        )

        file.close()

    def generate_bias(self, file_name: str, max_clauses: int, max_vars: int, max_body: int, task: str):
        file = open(file_name, "w")

        file.write(f"max_clauses({max_clauses}).\n"
                   f"max_vars({max_vars}).\n"
                   f"max_body({max_body}).\n")

        head_preds: List[BiasPredicate] = [BiasPredicate("precond", ["state"], ["in"], "head")]

        body_preds: List[BiasPredicate] = []

        body_preds.extend([
            # BiasPredicate("p0", ["state"], ["in"], "body"),
            BiasPredicate("cubelet", ["cubelet"], ["out"], "body"),
            BiasPredicate("has_stk", ["cubelet", "sticker"], ["in", "out"], "body"),

            BiasPredicate("match_0", ["state", "cubelet", "sticker"], ["in", "in", "in"], "body"),
            BiasPredicate("match_0_flip", ["state", "cubelet", "sticker"], ["in", "in", "in"], "body"),
            # BiasPredicate("match_cl", ["state", "cubelet", "sticker"], ["in", "in", "in"], "body"),
            # BiasPredicate("match_cc", ["state", "cubelet", "sticker"], ["in", "in", "in"], "body"),
            BiasPredicate("match_90", ["state", "cubelet", "sticker"], ["in", "in", "in"], "body"),
            BiasPredicate("match_180", ["state", "cubelet", "sticker"], ["in", "in", "in"], "body"),
            BiasPredicate("match_180_flip", ["state", "cubelet", "sticker"], ["in", "in", "in"], "body"),

            BiasPredicate("white", [], ["in"], "body"),
            BiasPredicate("yellow", [], ["in"], "body"),
            # BiasPredicate("orange", [], ["in"], "body"),
            # BiasPredicate("red", [], ["in"], "body"),
            # BiasPredicate("blue", [], ["in"], "body"),
            # BiasPredicate("green", [], ["in"], "body"),
        ])

        other_preds: List[BiasPredicate] = []

        file.write("\n% head and body\n")
        for bias_pred in head_preds + body_preds:
            head_body_str = bias_utils.body_head(bias_pred)
            file.write(f"{head_body_str}\n")

        file.write("\n% types\n")
        for bias_pred in head_preds + body_preds + other_preds:
            type_str = bias_utils.pred_type(bias_pred)
            if type_str is not None:
                file.write(f"{type_str}\n")

        file.write("\n% directions\n")
        for bias_pred in head_preds + body_preds + other_preds:
            direc_str = bias_utils.pred_direction(bias_pred)
            file.write(f"{direc_str}\n")

        file.write("\n% other constraints\n")

        file.write("\n% Variable helpers\n")
        file.write(
            "head_var(Clause, Var) :- head_literal(Clause, _, _, Vars), var_member(Var, Vars).\n"
            "body_var(Clause, Var) :- body_literal(Clause, _, _, Vars), var_member(Var, Vars).\n"
            "body_pred_vars(Clause, Pred, Vars) :- body_literal(Clause, Pred, _, Vars).\n"
            "head_pred_var(Clause, Pred, Var) :- head_literal(Clause, Pred, _, Vars), var_member(Var, Vars).\n"
            "body_pred_var(Clause, Pred, Var) :- body_literal(Clause, Pred, _, Vars), var_member(Var, Vars).\n")

        file.write("\n% Predicate types\n")
        file.write(
            "color_pred(white).\n"
            "color_pred(yellow).\n"
            "color_pred(orange).\n"
            "color_pred(red).\n"
            "color_pred(blue).\n"
            "color_pred(green).\n"
            
            "match_pred(match_0).\n"
            "match_pred(match_0_flip).\n"
            "match_pred(match_180).\n"
            "match_pred(match_180_flip).\n"
            "match_pred(match_cl).\n"
            "match_pred(match_cc).\n"
            "match_pred(match_90).\n"
        )

        file.write("\n%%% Cubelet constraints\n")

        # file.write("\n% Restrict number of variables\n")
        # file.write(":- #count{Vars: body_literal(C, cubelet, 1, Vars)} > 1.\n")

        """
        file.write("\n% Cubelet cannot have more than one sticker in a rel_pred with the same face\n")
        file.write(":- head_var(C, A), body_literal(C, cubelet, 1, (Cbl,)), body_literal(C, face, 1, (F,)), "
                   "#count{Stk: body_literal(C, P, 4, (A, Cbl, Stk, F)), rel_pred(P)} > 1.\n")

        # file.write("\n% onface cannot appear more than twice for the same cubelet\n")
        # file.write(":- body_pred_var(C, onface, V1), body_pred_var(C, onface, V2), "
        #           "#count{Vars: body_literal(C, onface, 4, Vars), Vars = (V1, V2, _, _)} > 2.\n")
        """

        file.write("\n%%% Sticker constraints\n")

        file.write("\n% Restrict number of variables (max two stk needed per cubelet)\n")
        file.write(":- body_literal(C, cubelet, 1, (Cbl,)), "
                   "#count{Stk: body_literal(C, has_stk, 2, (Cbl, Stk))} > 2.\n")

        file.write("\n% More than one sticker on the same cubelet cannot be the same color\n")
        file.write(":- clause(C), color_pred(P), body_literal(C, cubelet, 1, (Cbl,)), "
                   "#count{Stk: body_literal(C, P, 1, (Stk,)), body_literal(C, has_stk, 2, (Cbl, Stk))} > 1.\n")

        file.write("\n % Sticker can only be in one match_pred\n")
        file.write(":- clause(C), head_var(C, A), body_literal(C, has_stk, 2, (Cbl, Stk)), "
                   "#count{P: body_literal(C, P, 3, (A, Cbl, Stk)), match_pred(P)} > 1.\n")

        file.write("\n%%% Color constraints\n")

        file.write("\n% Only face and sticker in color pred\n")
        file.write(":- body_literal(C, P, 1, (Obj,)), color_pred(P), var_type(C, Obj, Type), "
                   "Type != face, Type != sticker.\n")

        file.write("\n% Object cannot be more than one color type\n")
        file.write(":- body_var(C, Obj), #count{P: body_literal(C, P, 1, (Obj,)), color_pred(P)} > 1.\n")

        """
        file.write("\n% Object cannot be in color_pred and match\n")
        file.write(":- body_literal(C, P_c, 1, (Obj,)), color_pred(P_c), "
                   "body_literal(C, P_m, 2, (Obj, _)), match_pred(P_m).\n")
        file.write(":- body_literal(C, P_c, 1, (Obj,)), color_pred(P_c), "
                   "body_literal(C, P_m, 2, (_, Obj)), match_pred(P_m).\n")

        file.write("\n% not_match only used with onface\n")
        file.write(":- head_var(C, A), "
                   "body_literal(C, has_stk, 2, (Cbl, Stk)), body_literal(C, not_match, 2, (Stk, F)), "
                   "#count{P: body_literal(C, P, 4, (A, Cbl, Stk, F)), rel_pred(P), P != onface} > 0.\n")
        """

        """
        file.write("\n% A sticker cannot be on more than one face\n")
        file.write(":- head_var(C, A), body_literal(C, cubelet, 1, (Cbl,)), body_literal(C, has_stk, 2, (Cbl, Stk)), "
                   "#count{F: body_literal(C, onface, 4, (A, Cbl, Stk, F))} > 1.\n")

        file.write("\n% A sticker cannot be opposite more than one face\n")
        file.write(":- head_var(C, A), body_literal(C, cubelet, 1, (Cbl,)), body_literal(C, has_stk, 2, (Cbl,Stk)), "
                   "#count{F: body_literal(C, face_rel_180, 4, (A, Cbl, Stk, F))} > 1.\n")

        file.write("\n% A sticker can only participate in one rel_pred\n")
        file.write(":- head_var(C, A), body_literal(C, cubelet, 1, (Cbl,)), body_literal(C, has_stk, 2, (Cbl, Stk)), "
                   "#count{F: body_literal(C, P, 4, (A, Cbl, Stk, F)), rel_pred(P)} > 1.\n")
        file.write(":- head_var(C, A), body_literal(C, cubelet, 1, (Cbl,)), body_literal(C, has_stk, 2, (Cbl, Stk)), "
                   "#count{P: body_literal(C, P, 4, (A, Cbl, Stk, _)), rel_pred(P)} > 1.\n")

        file.write("\n% Sticker cannot match with more than one face\n")
        file.write(":- body_literal(C, has_stk, 2, (_, Stk)), #count{F: body_literal(C, match, 2, (Stk, F))} > 1.\n")

        file.write("\n% Different stickers on the same cubelet cannot participate in match_pred with the same face\n")
        file.write(":- body_literal(C, cubelet, 1, (Cbl,)), body_literal(C, face, 1, (F,)), "
                   "#count{Stk: body_literal(C, has_stk, 2, (Cbl, Stk)), body_literal(C, P, 2, (Stk, F)), "
                   "match_pred(P)} > 1.\n")

        file.write("\n % Sticker cannot be in match and not_match at the same time\n")
        file.write(":- body_literal(C, match, 2, (Stk, F1)), body_literal(C, not_match, 2, (Stk, F2)).\n")

        file.write("\n % Sticker can only be in one match_pred\n")
        file.write(":- body_literal(C, has_stk, 2, (Cbl, Stk)), "
                   "#count{F: body_literal(C, P, 2, (Stk, F)), match_pred(P)} > 1.\n")
        file.write(":- body_literal(C, has_stk, 2, (Cbl, Stk)), "
                   "#count{P: body_literal(C, P, 2, (Stk, _)), match_pred(P)} > 1.\n")
        """

        """
        file.write("\n%%% Face constraints\n")

        file.write("\n% Restrict number of variables\n")
        file.write(":- #count{Vars: body_literal(C, face, 1, Vars)} > 3.\n")

        file.write("\n% Each face variable must reason about its color\n")
        file.write(":- body_literal(C, face, 1, (F,)), "
                   "#count{P_c: body_literal(C, P_c, 1, (F,)), color_pred(P_c)} = 0, "
                   "#count{P_m: body_literal(C, P_m, 2, (_, F)), match_pred(P_m)} = 0.\n")

        file.write("\n% Each face variable must participate in relation\n")
        file.write(":- body_literal(C, face, 1, (F,)), "
                   "#count{P: body_literal(C, P, 4, (_, _, _, F)), rel_pred(P)} = 0.\n")

        file.write("\n% If face in match_pred with stk must also be in rel with stk\n")
        file.write(":- head_var(C, A), body_literal(C, face, 1, (F,)), body_literal(C, has_stk, 2, (Cbl, Stk)), "
                   "body_literal(C, P_m, 2, (Stk, F)), match_pred(P_m), "
                   "#count{P: body_literal(C, P, 4, (A, Cbl, Stk, F)), rel_pred(P)} = 0.\n")

        file.write("\n% More than one face cannot be the same color\n")
        file.write(":- clause(C), color_pred(P), #count{F: body_literal(C, P, 1, (F,)), var_type(C, F, face)} > 1.\n")
        
        """

        # file.write("\n%%% Other constraints\n")
        # file.write(":- #count{Vars: body_literal(C, onface, 4, Vars)} > 1.\n")

        # file.write("\n%s onface with all variables but state the same is redundant\n")
        # file.write(":- #count{Clause: body_literal(Clause, onface, 4, (V1,V2,V3,V4)), "
        #           "body_literal(Clause, onface, 4, (V5,V6,V7,V8)), V1 != V5, V2 = V6, V3 = V7, V4 = V8} > 0.\n")

        file.close()

    def visualize(self, states: List[Cube3State]) -> np.ndarray:
        # initialize
        fig = plt.figure(figsize=(.64, .64))
        viz = InteractiveCube(3, self.generate_states(1, (0, 0))[0][0].colors)

        fig.add_axes(viz)
        canvas = FigureCanvas(fig)
        width, height = fig.get_size_inches() * fig.get_dpi()
        width = int(width)
        height = int(height)

        states_img: np.ndarray = np.zeros((len(states), 64, 64, 6))
        for state_idx, state in enumerate(states):
            # create image
            viz.new_state(state.colors)

            viz.set_rot(0)
            canvas.draw()
            image1 = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(width, height, 3) / 255

            viz.set_rot(1)
            canvas.draw()
            image2 = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(width, height, 3) / 255

            states_img[state_idx] = np.concatenate((image1, image2), axis=2)

        plt.close(fig)

        return states_img

    def get_macro_action(self, action_seq: List[int]) -> Cube3MacroAction:
        clause_orig: Clause = self._get_move_clause(action_seq)
        action_seq_syms: Set[Tuple[int, ...]] = set()

        use_symmetry: bool = True
        if use_symmetry:
            clauses: List[Clause] = []

            for sym_map in self.sym_maps:
                action_seq_sym: List[int] = []
                for action in action_seq:
                    action_sym: int = sym_map[action]
                    action_seq_sym.append(action_sym)

                if tuple(action_seq_sym) in action_seq_syms:
                    continue

                action_seq_syms.add(tuple(action_seq_sym))
                clause: Clause = self._get_move_clause(action_seq_sym)

                clauses.append(clause)

            prog: Program = Program(clauses)
        else:
            prog: Program = Program([clause_orig])
            action_seq_syms.add(tuple(action_seq))

        macro_action: Cube3MacroAction = Cube3MacroAction(action_seq_syms, prog, clause_orig)

        return macro_action

    def state_to_rep3d(self, state: Cube3State) -> np.ndarray:
        len_3d = self.cube_len + 2
        rep_3d: np.ndarray = np.zeros((len_3d, len_3d, len_3d), dtype=self.dtype)
        colors_np: np.ndarray = state.colors.copy().reshape((6, self.cube_len, self.cube_len))

        rep_3d[0, 1:(len_3d - 1), 1:(len_3d - 1)] = colors_np[0]  # white
        rep_3d[-1, 1:(len_3d - 1), 1:(len_3d - 1)] = np.fliplr(colors_np[1])  # yellow
        rep_3d[1:(len_3d - 1), 0, 1:(len_3d - 1)] = np.fliplr(np.rot90(colors_np[2], k=1))  # orange
        rep_3d[1:(len_3d - 1), -1, 1:(len_3d - 1)] = np.rot90(colors_np[3])  # red
        rep_3d[1:(len_3d - 1), 1:(len_3d - 1), -1] = np.fliplr(np.rot90(colors_np[4], k=1))  # blue
        rep_3d[1:(len_3d - 1), 1:(len_3d - 1), 0] = np.rot90(colors_np[5], k=1)  # green

        return rep_3d

    def rep3d_to_state(self, rep_3d: np.ndarray) -> Cube3State:
        len_3d = self.cube_len + 2

        colors_np = np.zeros((6, self.cube_len, self.cube_len), dtype=self.dtype)

        colors_np[0] = rep_3d[0, 1:(len_3d - 1), 1:(len_3d - 1)]  # white
        colors_np[1] = np.fliplr(rep_3d[-1, 1:(len_3d - 1), 1:(len_3d - 1)])  # yellow
        colors_np[2] = np.rot90(np.fliplr(rep_3d[1:(len_3d - 1), 0, 1:(len_3d - 1)]), k=3)  # orange
        colors_np[3] = np.rot90(rep_3d[1:(len_3d - 1), -1, 1:(len_3d - 1)], k=3)  # red
        colors_np[4] = np.rot90(np.fliplr(rep_3d[1:(len_3d - 1), 1:(len_3d - 1), -1]), k=3)  # blue
        colors_np[5] = np.rot90(rep_3d[1:(len_3d - 1), 1:(len_3d - 1), 0], k=3)  # green

        colors_np = colors_np.reshape(6 * self.cube_len * self.cube_len)
        state = Cube3State(colors_np.copy())

        return state

    def test_rep3d_state(self):
        for action_idx, action_str in enumerate(self.atomic_actions):
            state: Cube3State = Cube3State(np.arange(0, 54, 1, dtype=self.dtype))
            rep_3d = self.state_to_rep3d(state)

            face: str = action_str[0]
            act_dir: int = int(action_str[1:])

            if act_dir == 1:
                k_move: int = 3
            elif act_dir == -1:
                k_move: int = 1
            else:
                raise ValueError(f"Unknown action direction {act_dir}")

            rep_3d_new = rep_3d.copy()
            if face == "U":
                rep_3d_new[0, :, :] = np.rot90(rep_3d[0, :, :], k=k_move)
                rep_3d_new[1, :, :] = np.rot90(rep_3d[1, :, :], k=k_move)
            elif face == "D":
                rep_3d_new[-1, :, :] = np.rot90(rep_3d[-1, :, :], k=(k_move + 2) % 4)
                rep_3d_new[-2, :, :] = np.rot90(rep_3d[-2, :, :], k=(k_move + 2) % 4)
            elif face == "L":
                rep_3d_new[:, 0, :] = np.rot90(rep_3d[:, 0, :], k=(k_move + 2) % 4)
                rep_3d_new[:, 1, :] = np.rot90(rep_3d[:, 1, :], k=(k_move + 2) % 4)
            elif face == "R":
                rep_3d_new[:, -1, :] = np.rot90(rep_3d[:, -1, :], k=k_move)
                rep_3d_new[:, -2, :] = np.rot90(rep_3d[:, -2, :], k=k_move)
            elif face == "F":
                rep_3d_new[:, :, 0] = np.rot90(rep_3d[:, :, 0], k=k_move)
                rep_3d_new[:, :, 1] = np.rot90(rep_3d[:, :, 1], k=k_move)
            elif face == "B":
                rep_3d_new[:, :, -1] = np.rot90(rep_3d[:, :, -1], k=(k_move + 2) % 4)
                rep_3d_new[:, :, -2] = np.rot90(rep_3d[:, :, -2], k=(k_move + 2) % 4)

            state_from_3d = self.rep3d_to_state(rep_3d)
            state_from_3d_next = self.rep3d_to_state(rep_3d_new)

            state_next = self.next_state([state], [action_idx])[0][0]

            assert state == state_from_3d, f"rot_3d error for 3d conversion"
            assert state_next == state_from_3d_next, f"rot_3d error for {action_str}"
            # if state != state_from_3d:
            #    print(f"rot_3d error for 3d conversion")

            # if state_next != state_from_3d_next:
            #    print(f"rot_3d error for {action_str}")

    def get_symmetry(self, state: Cube3State, sym_type: str, face: str) -> Cube3State:
        if face.upper() == "U":
            axis = 0
        elif face.upper() == "L":
            axis = 1
        elif face.upper() == "F":
            axis = 2
        else:
            raise ValueError(f"Unknown rotate cube face {face}. Must be U, L, or F.")

        rep_3d = self.state_to_rep3d(state)
        if sym_type.upper() == "ROTATE_CUBE":
            axes = tuple([x for x in range(0, 3) if x != axis])
            rep_3d = np.rot90(rep_3d, k=3, axes=axes)
        elif sym_type.upper() == "MIRROR_CUBE":
            rep_3d = np.flip(rep_3d, axis=axis)
        else:
            raise ValueError(f"Unknwon symmetry type {sym_type}")

        state_sym: Cube3State = self.rep3d_to_state(rep_3d)
        return state_sym

    def get_sym_maps(self) -> List[Dict[int, int]]:
        stk_per_face: int = self.cube_len * self.cube_len
        state_np_orig = np.arange(0, stk_per_face * 6, 1, dtype=self.dtype) // stk_per_face
        state_np_orig = state_np_orig.astype(self.dtype)

        state: Cube3State = Cube3State(state_np_orig.copy())

        seen_states: Set[(Cube3State, bool)] = {(state, False)}
        states_curr: List[Tuple[State, bool]] = [(state, False)]

        while len(states_curr) > 0:
            states_curr_new: List[Tuple[State, bool]] = []
            for state, mirror in states_curr:
                for face in ["U", "L", "F"]:
                    for sym_type in ["rotate_cube", "mirror_cube"]:
                        state_next = self.get_symmetry(state, sym_type, face)
                        if sym_type == "mirror_cube":
                            mirror_next: bool = not mirror
                        else:
                            mirror_next = mirror
                        if (state_next, mirror_next) not in seen_states:
                            seen_states.add((state_next, mirror_next))
                            states_curr_new.append((state_next, mirror_next))

            states_curr = states_curr_new

        # sym map
        sym_maps: List[Dict[int, int]] = []
        for state, mirror in seen_states:
            sym_map: Dict[int, int] = dict()
            state_np = state.colors

            for face_num in range(6):
                action_orig: int = face_num * 2
                action_new: int = state_np[face_num * stk_per_face] * 2
                if mirror:
                    sym_map[action_orig] = action_new + 1
                    sym_map[action_orig + 1] = action_new
                else:
                    sym_map[action_orig] = action_new
                    sym_map[action_orig + 1] = action_new + 1

            sym_maps.append(sym_map)

        return sym_maps

    def get_num_symmetries(self):
        state: Cube3State = Cube3State(np.arange(0, 54, 1, dtype=self.dtype))

        seen_states: Set[Cube3State] = {state}
        states_curr: List[State] = [state]

        while len(states_curr) > 0:
            states_curr_new: List[State] = []
            for state in states_curr:
                for face in ["U", "L", "F"]:
                    for sym_type in ["rotate_cube", "mirror_cube"]:
                        state_next = self.get_symmetry(state, sym_type, face)
                        if state_next not in seen_states:
                            seen_states.add(state_next)
                            states_curr_new.append(state_next)

            states_curr = states_curr_new

    def _get_move_clause(self, action_seq: List[int]) -> Clause:
        head = Literal("act", ("A", "B"))
        body: List[Literal] = []

        for action_idx, action in enumerate(action_seq):
            if action_idx == 0:
                state_in: str = "A"
            else:
                state_in: str = f"S{action_idx}"

            if action_idx == (len(action_seq) - 1):
                state_out: str = "B"
            else:
                state_out: str = f"S{action_idx + 1}"

            face: str = self.faces[action // 2]
            if action % 2 == 0:
                direc: str = "cc"
            else:
                direc: str = "cl"

            act_logic: Literal = Literal("move", (f"{state_in}", face, direc, f"{state_out}"))
            body.append(act_logic)

        clause: Clause = Clause(head, body)

        return clause

    def _cubelet_to_name(self, cubelet: List[int]) -> str:
        face_idxs = np.floor(np.sort(cubelet) // (self.cube_len ** 2)).astype(np.int)
        cubelet_name = "".join([self.faces[idx] for idx in face_idxs])

        return cubelet_name

    def _move_np(self, states_np: np.ndarray, action: int) -> Tuple[np.ndarray, List[float]]:
        states_next_np: np.ndarray = states_np.copy()

        actions = [action]

        for action_part in actions:
            action_str: str = self.atomic_actions[action_part]
            states_next_np[:, self.rotate_idxs_new[action_str]] = states_next_np[:, self.rotate_idxs_old[action_str]]

        transition_costs: List[float] = [1.0 for _ in range(states_np.shape[0])]

        return states_next_np, transition_costs

    def _get_adj(self) -> None:
        # WHITE:0, YELLOW:1, BLUE:2, GREEN:3, ORANGE: 4, RED: 5
        self.adj_faces: Dict[int, np.ndarray] = {0: np.array([2, 5, 3, 4]),
                                                 1: np.array([2, 4, 3, 5]),
                                                 2: np.array([0, 4, 1, 5]),
                                                 3: np.array([0, 5, 1, 4]),
                                                 4: np.array([0, 3, 1, 2]),
                                                 5: np.array([0, 2, 1, 3])
                                                 }

    def _compute_rotation_idxs(self, cube_len: int,
                               moves: List[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        rotate_idxs_new: Dict[str, np.ndarray] = dict()
        rotate_idxs_old: Dict[str, np.ndarray] = dict()

        for move in moves:
            f: str = move[0]
            sign: int = int(move[1:])

            rotate_idxs_new[move] = np.array([], dtype=int)
            rotate_idxs_old[move] = np.array([], dtype=int)

            colors = np.zeros((6, cube_len, cube_len), dtype=np.int64)
            colors_new = np.copy(colors)

            # WHITE:0, YELLOW:1, BLUE:2, GREEN:3, ORANGE: 4, RED: 5

            adj_idxs = {0: {2: [range(0, cube_len), cube_len - 1], 3: [range(0, cube_len), cube_len - 1],
                            4: [range(0, cube_len), cube_len - 1], 5: [range(0, cube_len), cube_len - 1]},
                        1: {2: [range(0, cube_len), 0], 3: [range(0, cube_len), 0], 4: [range(0, cube_len), 0],
                            5: [range(0, cube_len), 0]},
                        2: {0: [0, range(0, cube_len)], 1: [0, range(0, cube_len)],
                            4: [cube_len - 1, range(cube_len - 1, -1, -1)], 5: [0, range(0, cube_len)]},
                        3: {0: [cube_len - 1, range(0, cube_len)], 1: [cube_len - 1, range(0, cube_len)],
                            4: [0, range(cube_len - 1, -1, -1)], 5: [cube_len - 1, range(0, cube_len)]},
                        4: {0: [range(0, cube_len), cube_len - 1], 1: [range(cube_len - 1, -1, -1), 0],
                            2: [0, range(0, cube_len)], 3: [cube_len - 1, range(cube_len - 1, -1, -1)]},
                        5: {0: [range(0, cube_len), 0], 1: [range(cube_len - 1, -1, -1), cube_len - 1],
                            2: [cube_len - 1, range(0, cube_len)], 3: [0, range(cube_len - 1, -1, -1)]}
                        }
            face_dict = {'U': 0, 'D': 1, 'L': 2, 'R': 3, 'B': 4, 'F': 5}
            face = face_dict[f]

            faces_to = self.adj_faces[face]
            if sign == 1:
                faces_from = faces_to[(np.arange(0, len(faces_to)) + 1) % len(faces_to)]
            else:
                faces_from = faces_to[(np.arange(len(faces_to) - 1, len(faces_to) - 1 + len(faces_to))) % len(faces_to)]

            cubes_idxs = [[0, range(0, cube_len)], [range(0, cube_len), cube_len - 1],
                          [cube_len - 1, range(cube_len - 1, -1, -1)], [range(cube_len - 1, -1, -1), 0]]
            cubes_to = np.array([0, 1, 2, 3])
            if sign == 1:
                cubes_from = cubes_to[(np.arange(len(cubes_to) - 1, len(cubes_to) - 1 + len(cubes_to))) % len(cubes_to)]
            else:
                cubes_from = cubes_to[(np.arange(0, len(cubes_to)) + 1) % len(cubes_to)]

            for i in range(4):
                idxs_new = [[idx1, idx2] for idx1 in np.array([cubes_idxs[cubes_to[i]][0]]).flatten() for idx2 in
                            np.array([cubes_idxs[cubes_to[i]][1]]).flatten()]
                idxs_old = [[idx1, idx2] for idx1 in np.array([cubes_idxs[cubes_from[i]][0]]).flatten() for idx2 in
                            np.array([cubes_idxs[cubes_from[i]][1]]).flatten()]
                for idxNew, idxOld in zip(idxs_new, idxs_old):
                    flat_idx_new = np.ravel_multi_index((face, idxNew[0], idxNew[1]), colors_new.shape)
                    flat_idx_old = np.ravel_multi_index((face, idxOld[0], idxOld[1]), colors.shape)
                    rotate_idxs_new[move] = np.concatenate((rotate_idxs_new[move], [flat_idx_new]))
                    rotate_idxs_old[move] = np.concatenate((rotate_idxs_old[move], [flat_idx_old]))

            # Rotate adjacent faces
            face_idxs = adj_idxs[face]
            for i in range(0, len(faces_to)):
                face_to = faces_to[i]
                face_from = faces_from[i]
                idxs_new = [[idx1, idx2] for idx1 in np.array([face_idxs[face_to][0]]).flatten() for idx2 in
                            np.array([face_idxs[face_to][1]]).flatten()]
                idxs_old = [[idx1, idx2] for idx1 in np.array([face_idxs[face_from][0]]).flatten() for idx2 in
                            np.array([face_idxs[face_from][1]]).flatten()]
                for idxNew, idxOld in zip(idxs_new, idxs_old):
                    flat_idx_new = np.ravel_multi_index((face_to, idxNew[0], idxNew[1]), colors_new.shape)
                    flat_idx_old = np.ravel_multi_index((face_from, idxOld[0], idxOld[1]), colors.shape)
                    rotate_idxs_new[move] = np.concatenate((rotate_idxs_new[move], [flat_idx_new]))
                    rotate_idxs_old[move] = np.concatenate((rotate_idxs_old[move], [flat_idx_old]))

        return rotate_idxs_new, rotate_idxs_old
