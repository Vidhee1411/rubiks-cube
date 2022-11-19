from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Optional
import torch.nn as nn
from popper.core import Program


class State(ABC):
    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass


class MacroAction(ABC):
    @abstractmethod
    def get_macro_action(self) -> Program:
        pass

    @abstractmethod
    def set_precond(self, precond: Program):
        pass

    @abstractmethod
    def get_precond(self) -> Optional[Program]:
        pass

    @abstractmethod
    def get_complexity(self) -> float:
        pass

    @abstractmethod
    def to_string(self) -> str:
        pass

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass


class Environment(ABC):
    def __init__(self):
        self.dtype = np.float
        self.fixed_actions: bool = True

    @property
    @abstractmethod
    def num_actions_max(self) -> int:
        pass

    @property
    @abstractmethod
    def num_subgoals(self) -> int:
        pass

    @abstractmethod
    def next_state(self, states: List[State], actions: List[int]) -> Tuple[List[State], List[float]]:
        """ Get the next state and transition cost given the current state and action

        @param states: List of states
        @param actions: Actions to take
        @return: Next states, transition costs
        """
        pass

    @abstractmethod
    def rand_action(self, states: List[State]) -> List[int]:
        """ Get random actions that could be taken in each state

        @param states: List of states
        @return: List of random actions
        """
        pass

    @abstractmethod
    def generate_goal_states(self, num_states: int) -> List[State]:
        """ Generate goal states

        @param num_states: Number of states to generate
        @return: List of states
        """
        pass

    @abstractmethod
    def is_solved(self, states: List[State], states_goal: List[State]) -> np.array:
        """ Returns whether or not state is solved

        @param states: List of states
        @param states_goal: List of goal states
        @return: Boolean numpy array where the element at index i corresponds to whether or not the
        state at index i is solved
        """
        pass

    @abstractmethod
    def state_to_nnet_input(self, states: List[State]) -> List[np.ndarray]:
        """ State to numpy arrays to be fed to the neural network

        @param states: List of states
        @return: List of numpy arrays. Each index along the first dimension of each array corresponds to the index of
        a state.
        """
        pass

    def expand(self, states: List[State]) -> Tuple[List[List[State]], List[np.ndarray]]:
        """ Generate all children for the state
        @param states: List of states
        @return: Children of each state, Transition costs for each state
        """
        pass

    @abstractmethod
    def get_nnet(self) -> nn.Module:
        """ Get the neural network model for the dqn

        @return: neural network model
        """
        pass

    @abstractmethod
    def state_to_predicate(self, states: List[State]) -> List[str]:
        pass

    @abstractmethod
    def predicate_to_state(self, predicates: List[str]) -> List[State]:
        pass

    @abstractmethod
    def generate_bk(self, file: str):
        pass

    @abstractmethod
    def generate_bias(self, file: str, max_clauses: int, max_vars: int, max_body: int, task: str):
        pass

    @abstractmethod
    def visualize(self, states: List[State]) -> np.ndarray:
        pass

    @abstractmethod
    def generate_states(self, num_states: int, backwards_range: Tuple[int, int]) -> Tuple[List[State], List[int]]:
        pass

    @abstractmethod
    def generate_focused_start_goals(self, states: List[State]) -> Tuple[List[State], List[State]]:
        pass

    @abstractmethod
    def get_macro_action(self, action_seq: List[int]) -> MacroAction:
        pass
