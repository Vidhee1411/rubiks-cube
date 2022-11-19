import pdb
from typing import List, Tuple, Set, Optional
import argparse

from environments.environment_abstract import Environment, State, MacroAction
from utils import nnet_utils, program_utils, data_utils, viz_utils
from environments.cube3 import Cube3
from popper.tester import Tester

from search_methods.astar import AStar, get_path

from utils import popper_utils
from utils.prolog_utils import PrologProc, using_prog
from popper.core import Program

from nlg_pred_gen.pred_description_gen import pred_description_generation
from utils.nlg_utils import get_precondition, preprocess_precondition

import os
import sys
import time
import numpy as np
import random
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('kbpath', help='Path to the knowledge base one wants to learn on')
    parser.add_argument('--env', type=str, default="cube3", help='')
    parser.add_argument('--redo', default=False, action='store_true', help='Start from beginning')

    # solution paths
    parser.add_argument('--heur', type=str, default="saved_models/cube3m/current/",
                        help='Location of heuristic function')
    parser.add_argument('--num_states', type=int, default=1000, help='Number of states to use to generate explanation.')
    parser.add_argument('--max_states', type=int, default=1000000, help='Max number of states.')
    parser.add_argument('--num_viz', type=int, default=0, help='If greater than 0, output file not written')

    parser.add_argument('--ilp_v', default=False, action='store_true', help='Verbose for ILP')
    parser.add_argument('--astar_v', default=False, action='store_true', help='Verbose for A* search')
    parser.add_argument('--debug', default=False, action='store_true', help='Do not log data to make pdb easier')

    return parser.parse_args()


def get_m_acts_focused(states_start: List[State], env: Environment, heuristic_fn, prolog: PrologProc,
                       macro_actions_ban: Set[MacroAction], astar_verbose: bool, viz: bool) -> List[MacroAction]:
    # initialize
    # macro_actions_prev: Set[MacroAction] = macro_actions_learned.union(macro_actions_ban)
    macro_action_set: Set[MacroAction] = set()

    # Get paths for macro actions
    states_start, states_focused = env.generate_focused_start_goals(states_start)
    if viz > 0:
        viz_utils.visualize_examples(env, [states_start[0], states_focused[0]])

    astar = AStar(states_start, states_focused, env, heuristic_fn, weights=[1.0] * len(states_start),
                  macro_actions_forbid=macro_actions_ban)
    while not min(astar.has_found_goal()):
        astar.step(heuristic_fn, 1, verbose=astar_verbose)

    # find all macro actions
    states_start_p: List[str] = env.state_to_predicate(states_start)
    states_focused_p: List[str] = env.state_to_predicate(states_focused)
    for state_idx in range(len(states_start)):
        # ensure transition is salient
        state_start_p: str = states_start_p[state_idx]
        state_focused_p: str = states_focused_p[state_idx]

        res = prolog.query(f"salient_start({state_start_p}), salient_t({state_start_p},{state_focused_p})")
        found_salient: bool = len(res) > 0
        assert found_salient, f"state goal pair number {state_idx} is not salient, but should be"

        # get macro action
        goal_node = astar.get_cheapest_goal_node(state_idx)
        _, action_path, _ = get_path(goal_node)

        macro_action: MacroAction = env.get_macro_action(action_path)
        macro_action_set.add(macro_action)
        # if macro_action not in macro_actions_prev:
        #    macro_action_set.add(macro_action)

        assert len(action_path) > 0, "Length of macro actions should be greater than 0"

    # get simplest macro action
    macro_actions: List[MacroAction] = list(macro_action_set)
    print(f"Macro actions found: {len(macro_actions)}")

    return macro_actions


def get_precond_train_data(macro_action: MacroAction, states_p: List[str],
                           prolog: PrologProc) -> Tuple[List[str], List[str]]:
    pos_idxs: List[int] = []
    neg_idxs: List[int] = []
    with using_prog(prolog, macro_action.get_macro_action()):
        for state_idx, state_p in enumerate(states_p):
            res = prolog.query(f"act({state_p}, B), salient_t({state_p}, B), !")

            if len(res) > 0:
                pos_idxs.append(state_idx)
            else:
                neg_idxs.append(state_idx)

    pos_exs_p = [f"precond({states_p[idx]})" for idx in pos_idxs]
    neg_exs_p = [f"precond({states_p[idx]})" for idx in neg_idxs]

    return pos_exs_p, neg_exs_p


def get_precond_prog(pos_exs: List[str], neg_exs: List[str], env: Environment, kbpath: str,
                     debug: bool = True) -> Optional[Program]:
    # initialize
    bk_file_name = os.path.join(kbpath, 'bk.pl')
    bias_file_name = os.path.join(kbpath, 'bias.pl')

    env.generate_bk(bk_file_name)
    max_clauses: int = 1
    max_vars: int = 4
    max_body: int = 8
    env.generate_bias(bias_file_name, max_clauses, max_vars, max_body, "precond")

    # get precond program
    partial: bool = True
    tester = Tester(kbpath, pos_exs, neg_exs, test_all_pos=partial)
    precond_prog, _ = popper_utils.popper(bias_file_name, tester, max_body, partial=partial, debug=debug, stats=debug)
    tester.close()

    if debug:
        print("")

    return precond_prog


def apply_macro_action(state_p: str, prolog: PrologProc, macro_action: MacroAction) -> Tuple[Optional[str], bool]:
    state_next_p: Optional[str] = None
    with using_prog(prolog, macro_action.get_precond()):
        precond_res = prolog.query(f"precond({state_p})")

    if len(precond_res) > 0:
        with using_prog(prolog, macro_action.get_macro_action()):
            act_res = prolog.query(f"act({state_p}, B), salient_t({state_p}, B)")

        states_next_p_l: List[str] = list(set([x['B'] for x in act_res]))
        if len(states_next_p_l) == 0:
            print("ERROR:")
            print(f"Macro action:\n{macro_action.to_string()}")
            print("Precondition:")
            program_utils.print_program(macro_action.get_precond())
            raise AssertionError("Macro action says it is applicable, but it is not")

        state_next_p: str = random.choice(states_next_p_l)
        state_moved = True
    else:
        state_moved: bool = False

    return state_next_p, state_moved


def apply_macro_actions(states_p: List[str], prolog: PrologProc,
                        macro_actions: List[MacroAction]) -> Tuple[List[str], List[bool], List[Tuple[str, int]]]:
    states_moved: List[bool] = [False for _ in range(len(states_p))]
    move_data: List[Tuple[str, int]] = []

    for state_idx, state_p in enumerate(states_p):
        macro_action_idx: int = 0
        while (macro_action_idx < len(macro_actions)) and (len(prolog.query(f"solved({state_p})")) == 0):
            macro_action: MacroAction = macro_actions[macro_action_idx]
            state_next_p, state_moved = apply_macro_action(state_p, prolog, macro_action)
            states_moved[state_idx] = states_moved[state_idx] | state_moved

            if state_moved:
                move_data.append((state_p, macro_action_idx))
                state_p = state_next_p
                macro_action_idx = 0
            else:
                macro_action_idx += 1

        states_p[state_idx] = state_p

    return states_p, states_moved, move_data


def generate_states_p(env: Environment, num_states: int) -> List[str]:
    states_unsolved, _ = env.generate_states(num_states, (100, 200))
    states_goal: List[State] = env.generate_goal_states(num_states)  # TODO make one function
    is_solved = env.is_solved(states_unsolved, states_goal)
    states_unsolved = [x for x, y in zip(states_unsolved, is_solved) if y == 0]
    states_unsolved_p = env.state_to_predicate(states_unsolved)

    return states_unsolved_p


def get_m_acts(env: Environment, kbpath: str, states_p: List[str], is_solved,
               m_acts_learned: List[MacroAction], m_acts_ban: Set[MacroAction], prolog: PrologProc,
               heuristic_fn, itr: int, save_file: str, num_viz: int, astar_v: bool, ilp_v: bool):
    while min(is_solved) is False:
        print(f"******* Iteration: {itr} *******")
        # get macro action
        print("---Getting Macro Actions---")
        start_time = time.time()
        print(f"Number of banned macro actions: {len(m_acts_ban)}")
        states_unsolved_p = [x for x, y in zip(states_p, is_solved) if y is False]
        states_unsolved = env.predicate_to_state(states_unsolved_p)

        m_acts: List[MacroAction] = get_m_acts_focused(states_unsolved, env, heuristic_fn, prolog, m_acts_ban,
                                                       astar_v, False)
        if len(m_acts) == 0:
            raise Exception(f"No macro actions found")

        m_act_complexity: List[float] = [x.get_complexity() for x in m_acts]
        min_idx: int = int(np.argmin(m_act_complexity))
        # import pdb
        # pdb.set_trace()
        m_act: MacroAction = m_acts[min_idx]
        print(f"Macro action chosen:\n{m_act.to_string()}")
        print(f"Time: {time.time() - start_time}\n")

        precond_prog: Optional[Program] = None
        if m_act is not None:
            # get training data
            print("---Getting Training Data---")
            start_time = time.time()
            pos_exs_p, neg_exs_p = get_precond_train_data(m_act, states_unsolved_p, prolog)
            assert len(pos_exs_p) > 0, "Number of positive examples should be greater than 1"
            print(f"Positive: {len(pos_exs_p)}, Negative: {len(neg_exs_p)}, Total: {len(pos_exs_p) + len(neg_exs_p)} "
                  f"(Time: {time.time() - start_time})\n")
            if num_viz > 0:
                pos_exs = env.predicate_to_state(pos_exs_p)
                neg_exs = env.predicate_to_state(neg_exs_p)
                viz_utils.visualize_examples(env, pos_exs[:num_viz] + neg_exs[:num_viz])

            # attempt to learn precondition program
            print(f"---Learning precondition program---")
            start_time = time.time()
            precond_prog: Optional[Program] = get_precond_prog(pos_exs_p, neg_exs_p, env, kbpath, debug=ilp_v)

            print(f"Result for macro action:\n{m_act.to_string()}")
            if precond_prog is None:
                print("Precondition not learned")
                m_acts_ban.add(m_act)
                print(f"Time: {time.time() - start_time}\n")
            else:
                print("Precondition:")
                # pdb.set_trace()
                program_utils.print_program(precond_prog)
                m_act.set_precond(precond_prog)
                precondition_list = get_precondition(m_act)
                predicate_list = preprocess_precondition(precondition_list)
                generated_description = pred_description_generation(predicate_list)
                print("Description:")
                print(generated_description[0])
                m_acts_learned.append(m_act)
                m_acts_ban = set()
                print(f"Time: {time.time() - start_time}\n")


        if (precond_prog is not None) or (m_act is None):
            # move states
            print(f"---Applying {len(m_acts_learned)} macro action(s) to {len(states_unsolved_p)} unsolved "
                  f"states---")
            start_time = time.time()
            states_unsolved_p, states_moved, _ = apply_macro_actions(states_unsolved_p, prolog, m_acts_learned)

            is_solved_moved = [len(prolog.query(f"solved({x})")) > 0 for x in states_unsolved_p]
            print(f"Num moved: {sum(states_moved)}, Num Solved: {sum(is_solved_moved)}")

            # update states
            for idx_unsolved, idx in enumerate(np.where(np.array(is_solved) == 0)[0]):
                state_unsolved_p: str = states_unsolved_p[idx_unsolved]
                is_solved[idx] = len(prolog.query(f"solved({state_unsolved_p})")) > 0
                states_p[idx] = state_unsolved_p

            print(f"Time: {time.time() - start_time}\n")

        # print progress
        print("---Progress---")
        print(f"Macro actions learned: {len(m_acts_learned)}")
        print(f"{np.sum(is_solved)}/{len(states_p)} states solved ({100 * np.sum(is_solved) / len(states_p)}%)")
        print("")

        pickle.dump((states_p, is_solved, m_acts_learned, m_acts_ban, itr),
                    open(save_file, "wb"), protocol=-1)

        itr += 1

    return m_acts_learned


def induce_dt(env: Environment, num_states: int, m_acts_learned: List[MacroAction],
              prolog: PrologProc, save_file: str):
    print("---Getting training data---")
    start_time = time.time()
    states_p = generate_states_p(env, num_states)

    train_data: List[Tuple[str, int]]
    _, _, train_data = apply_macro_actions(states_p, prolog, m_acts_learned)
    is_solved: List[bool] = [len(prolog.query(f"solved({x})")) > 0 for x in states_p]
    assert min(is_solved), "Macro actions should solve all states"
    print("Time: %s\n" % (time.time() - start_time))


def main():
    # parse arguments
    args = parse_args()
    if not os.path.exists(args.kbpath):
        os.makedirs(args.kbpath)
    if (not args.debug) and (not args.num_viz):
        output_file: str = f"{args.kbpath}/output.txt"
        if args.redo:
            mode: str = "w"
        else:
            mode: str = "a"
        sys.stdout = data_utils.Logger(output_file, mode)

    env: Environment = Cube3(1)

    # initialize
    print("---Initializing Prolog---")
    start_time = time.time()
    salient_start: str = "salient_start(A) :- true"
    salient_transition: str = "salient_t(A,B) :- not_in_place(A,Cbl),in_place(B,Cbl),in_place_subset(A,B), !"
    print(f"Salient start state:\n{salient_start}")
    print(f"Salient transition:\n{salient_transition}")

    bk_file_name = os.path.join(args.kbpath, 'bk.pl')
    env.generate_bk(bk_file_name)

    prolog: PrologProc = PrologProc()
    prolog.consult(bk_file_name)
    prolog.assertz(salient_transition)
    prolog.assertz(salient_start)
    print("Time: %s\n" % (time.time() - start_time))

    # load heuristic function
    print("---Loading heuristic function---")
    start_time = time.time()
    device, devices, on_gpu = nnet_utils.get_device()
    heuristic_fn = nnet_utils.load_heuristic_fn("saved_models/cube3m/current/", device, on_gpu, env.get_nnet(),
                                                env, clip_zero=True, batch_size=10000)

    print("Time: %s\n" % (time.time() - start_time))

    save_file: str = f"{args.kbpath}/progress.pkl"
    if os.path.isfile(save_file) and not args.redo:
        print("---Loading Data---")
        start_time = time.time()
        states_p, is_solved, m_acts_learned, m_acts_ban, itr = pickle.load(open(save_file, "rb"))
        # print("--------------------")
        # print("--------------------")
        # print("--------------------")
        # print("--------------------")
        # print("--------------------")
        # print(states_p[0])
        # print("--------------------")
        # print("--------------------")
        # print("--------------------")
        # print("--------------------")
        # print("--------------------")
        # print(is_solved)
        # print("--------------------")
        # print("--------------------")
        # print("--------------------")
        # print("--------------------")
        # print("--------------------")
        # print(m_acts_learned)
        # print("--------------------")
        # print("--------------------")
        # print("--------------------")
        # print("--------------------")
        # print("--------------------")
        # print(m_acts_ban)
        # print("--------------------")
        # print("--------------------")
        # print("--------------------")
        # print("--------------------")
        # print("--------------------")
        # print(itr)
        # print("--------------------")
        # print("--------------------")
        # print("--------------------")
        # print("--------------------")
        # print("--------------------")

        # exit(0)
        print("Time: %s\n" % (time.time() - start_time))
    else:
        # get start states
        print("---Getting Start States---")
        start_time = time.time()

        states_p = generate_states_p(env, args.num_states)
        is_solved: List[bool] = [len(prolog.query(f"solved({x})")) > 0 for x in states_p]

        print("Time: %s\n" % (time.time() - start_time))

        m_acts_learned: List[MacroAction] = []
        m_acts_ban: Set[MacroAction] = set()

        itr: int = 0

    # find macro actions
    print("---Finding Macro Actions---")
    start_time = time.time()
    m_acts_learned: List[MacroAction] = get_m_acts(env, args.kbpath, states_p, is_solved, m_acts_learned,
                                                   m_acts_ban, prolog, heuristic_fn, itr, save_file,
                                                   args.num_viz, args.astar_v, args.ilp_v)
    print("Macro action time: %s\n" % (time.time() - start_time))

    print("---Building Decision Tree---")
    start_time = time.time()
    save_file_dt: str = f"{args.kbpath}/decision_tree.pkl"

    induce_dt(env, args.num_states, m_acts_learned, prolog, save_file_dt)

    print("Decision tree time: %s\n" % (time.time() - start_time))

    prolog.close()


if __name__ == '__main__':
    main()
