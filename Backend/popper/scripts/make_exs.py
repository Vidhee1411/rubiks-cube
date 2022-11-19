from typing import List, Tuple, Set
from utils import nnet_utils
from environments.cube3 import Cube3, Cube3State
from search_methods.astar import AStar, Node, get_path
import numpy as np


def get_state_str(state: Cube3State, faces: List[str], env: Cube3):
    onfaces = []

    for cubelet in env.subgoal_cubelets[env.subgoal_num]:
        if cubelet in env.cbs_e:
            for cubelet_idx in cubelet:
                face_idx = int(np.floor(np.where(state.colors == cubelet_idx)[0][0] / 9))
                onfaces.append(faces[face_idx])

    onface_str = ", ".join(onfaces)
    state_str = f"state({onface_str})"

    return state_str


def main():
    faces: List[str] = ['w', 'y', 'o', 'r', 'b', 'g']
    action_names: List[Tuple[str, str]] = [(f, n) for f in ['w', 'y', 'o', 'r', 'b', 'g'] for n in ["cc", "cl"]]
    action_names = action_names + [(f, "op") for f in ['w', 'y', 'o', 'r', 'b', 'g']]

    print("Generating states")
    num_states: int = 100
    env: Cube3 = Cube3(0)
    states: List[Cube3State]
    states, _ = env.generate_states(num_states, (100, 200))
    states_goal = env.generate_goal_states(num_states)

    device, devices, on_gpu = nnet_utils.get_device()
    heuristic_fn = nnet_utils.load_heuristic_fn("saved_models/cube3m/current/", device, on_gpu, env.get_nnet(),
                                                env, clip_zero=True, batch_size=10000)

    # def heuristic_fn(x, _):
    #    return np.zeros(len(x))

    print("Doing search")
    astar = AStar(states, states_goal, env, heuristic_fn, weights=[1.0] * len(states))
    num_itrs: int = 0
    while not min(astar.has_found_goal()):
        astar.step(heuristic_fn, 1, verbose=True)
        num_itrs += 1

    print("Making examples")
    state_to_action: Set[Tuple[str, str, str, str]] = set()
    for inst_idx in range(len(states)):
        goal_node: Node = astar.get_cheapest_goal_node(inst_idx)
        path, actions, _ = get_path(goal_node)
        for path_idx in range(len(path)-1):
            state: Cube3State = path[path_idx]
            state_next: Cube3State = path[path_idx + 1]
            action: int = actions[path_idx]

            action_face, action_dir = action_names[action]

            state_str: str = get_state_str(state, faces, env)
            state_next_str: str = get_state_str(state_next, faces, env)

            state_to_action.add((state_str, action_face, action_dir, state_next_str))

    file = open("popper/examples/cube/exs.pl", "w")
    for state_prev, action_face, action_dir, state_curr in state_to_action:
        # file.write(f"pos(f({state})).\n")
        file.write(f"pos(move({state_prev}, {action_face}, {action_dir})).\n")
    file.write("\n")

    """
    for direc_neg in direcs:
        for face_neg in faces:
            file.write(f"neg(move({face_neg}, {direc_neg}, w_c, o_c)).\n")
    file.write("\n")
    """

    """
    file.write("\n")
    for face, direc, w_pos, o_pos in state_to_action:
        for direc_neg in direcs:
            for face_neg in faces:
                if (direc_neg == direc) and (face_neg == face):
                    continue

                file.write(f"neg(move({face_neg}, {direc_neg}, {w_pos}, {o_pos})).\n")
    """

    file.close()


if __name__ == '__main__':
    main()
