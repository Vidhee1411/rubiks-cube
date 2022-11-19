from typing import List, Tuple
import numpy as np
from environments.environment_abstract import Environment, State


def is_valid_soln(state: State, state_goal: State, soln: List[int], env: Environment) -> bool:
    state_soln: State = state
    move: int
    for move in soln:
        state_soln = env.next_state([state_soln], [move])[0][0]

    return env.is_solved([state_soln], [state_goal])[0]


def q_step(states: List, heuristic_fn, env: Environment, eps_l: List[float]) -> Tuple[np.array, List[int], List[State]]:
    # ctgs for each action
    ctg_acts = heuristic_fn(states)

    # get actions
    actions: List[int] = list(np.argmin(ctg_acts, axis=1))

    eps_rand_moves = np.random.random(len(eps_l)) < np.array(eps_l)

    rand_idxs: np.array = np.where(eps_rand_moves)[0]
    if rand_idxs.shape[0] > 0:
        num_actions_rand: np.array = np.array([len(ctg_acts[idx]) for idx in rand_idxs])
        num_actions_t_rand = (num_actions_rand - 1) * np.random.rand(rand_idxs.shape[0])

        num_actions_t_rand = np.round(num_actions_t_rand)
        for idx, action_rand in zip(rand_idxs, num_actions_t_rand):
            actions[idx] = int(action_rand)

    """
    for idx in range(len(actions)):
        if eps_rand_moves[idx]:
            num_actions: int = len(ctg_acts[idx])
            actions[idx] = np.random.choice(num_actions)
    """

    # take action
    states_next: List[State]
    tcs: List[float]
    states_next, tcs = env.next_state(states, actions)

    # min cost-to-go for next state
    ctg_acts_next = heuristic_fn(states_next)
    ctg_acts_next_max = ctg_acts_next.min(axis=1)

    # backup cost-to-go
    ctg_backups = np.array(tcs) + ctg_acts_next_max

    is_solved = env.is_solved(states)
    ctg_backups = ctg_backups * np.logical_not(is_solved)

    return ctg_backups, actions, states_next
