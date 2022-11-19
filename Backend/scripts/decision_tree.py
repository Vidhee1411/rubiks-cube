from typing import List, Dict
from sklearn import tree
import pandas as pd
from environments.environment_abstract import Environment, State
from search_methods.astar import AStar, Node, get_path
from utils import env_utils
from matplotlib import pyplot as plt
import numpy as np

from argparse import ArgumentParser


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--env', type=str, required=True, help="Environment")
    parser.add_argument('--num_states', type=int, required=True, help="")
    parser.add_argument('--astar_weight', type=float, default=0.0, help="")
    args = parser.parse_args()

    # get environment
    env: Environment = env_utils.get_environment(args.env, 0)

    # get heuristic
    def heuristic_fn(x, *_):
        return np.zeros((len(x), env.num_actions_max), dtype=np.float)

    # do A* search
    states, _ = env.generate_states(args.num_states, (0, 30))
    astar = AStar(states, env, heuristic_fn, weights=[args.astar_weight] * len(states))
    while not min(astar.has_found_goal()):
        astar.step(heuristic_fn, 1)

    # get states and actions
    states_train: List[State] = []
    actions_train: List[int] = []
    for inst_idx in range(len(astar.instances)):
        goal_node: Node = astar.get_cheapest_goal_node(inst_idx)
        states_path, actions_path, _ = get_path(goal_node)
        actions_path.append(-1)

        states_train.extend(states_path)
        actions_train.extend(actions_path)

    # do classification
    states_train_features: Dict = env.get_features(states_train)

    dt = tree.DecisionTreeClassifier()
    dt = dt.fit(states_train_features['features'], actions_train)

    print(tree.export_text(dt))

    plt.figure(figsize=(15, 15))
    tree_viz = tree.plot_tree(dt, feature_names=states_train_features['feature_names'],
                              class_names=states_train_features['target_names'], filled=False, impurity=False,
                              node_ids=False)

    plt.show()

    actions_pred = dt.predict(states_train_features['features'])
    print("Accuracy: %f" % (100 * np.mean(actions_pred == np.array(actions_train))))

    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    main()
