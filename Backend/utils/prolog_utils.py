from typing import List, Union

import pyswip
from pyswip import Prolog, PL_STRINGS_MARK
from torch.multiprocessing import get_context
import numpy as np
from contextlib import contextmanager

from popper.core import Program


def _prolog_proc(args_queue, return_queue):
    prolog = Prolog()
    while True:
        args = args_queue.get()
        if args is None:
            break
        elif args[0] == "assertz":
            prolog.assertz(args[1])
        elif args[0] == "retractall":
            prolog.retractall(args[1])
        elif args[0] == "consult":
            prolog.consult(args[1])
        elif args[0] == "query":
            with PL_STRINGS_MARK():
                ret = list(prolog.query(args[1]))

            for i in range(len(ret)):
                for key, val in ret[i].items():
                    ret[i][key] = pyswip_output_to_str(val)

            return_queue.put(ret)
        else:
            raise ValueError("Unknown prolog method %s" % args[0])


class PrologProc:
    def __init__(self):
        ctx = get_context("spawn")
        self.args_queue: ctx.Queue = ctx.Queue()
        self.return_queue: ctx.Queue = ctx.Queue()

        self.proc = ctx.Process(target=_prolog_proc, args=(self.args_queue, self.return_queue))
        self.proc.daemon = True
        self.proc.start()

    def assertz(self, assert_str: str):
        self.args_queue.put(("assertz", assert_str))

    def retractall(self, retract_str: str):
        self.args_queue.put(("retractall", retract_str))

    def consult(self, file_name: str) -> None:
        self.args_queue.put(("consult", file_name))

    def query(self, q: str) -> List:
        self.args_queue.put(("query", q))
        return self.return_queue.get()

    def get_in_place(self, state: str) -> np.array:
        """
        res: List[Dict] = self.query(f'human_heur({state}, H)')
        h_vals: List[int] = [x['H'] for x in res]
        import pdb
        pdb.set_trace()
        assert max(h_vals) == min(h_vals), "state should map to a one unique heuristic value"

        return h_vals[0]
        """
        # TODO automate

        in_place: np.array = np.zeros(4, dtype=np.int)
        for edge_idx, edge in enumerate(["wg", "wo", "wr", "wb"]):
            if len(self.query(f'in_place({state},{edge})')) > 0:
                in_place[edge_idx] = 1

        return in_place

    def get_num_added(self, state: str, state_next: str) -> int:
        num_added: int = 0
        for edge in ["wg", "wo", "wr", "wb"]:
            not_in_place: bool = len(self.query(f'not_in_place({state},{edge})')) > 0
            in_place_next: bool = len(self.query(f'in_place({state_next},{edge})')) > 0

            if not_in_place and in_place_next:
                num_added += 1

        return num_added

    def get_num_removed(self, state: str, state_next: str) -> int:
        num_added: int = 0
        for edge in ["wg", "wo", "wr", "wb"]:
            in_place: bool = len(self.query(f'in_place({state},{edge})')) > 0
            not_in_place_next: bool = len(self.query(f'not_in_place({state_next},{edge})')) > 0

            if in_place and not_in_place_next:
                num_added += 1

        return num_added

    def close(self):
        self.args_queue.put(None)
        self.proc.join()


@contextmanager
def using_prog(prolog: PrologProc, program: Program):
    current_clauses = set()
    try:
        for clause in program.clauses:
            prolog.assertz(clause.to_code())

            current_clauses.add((clause.head.predicate, clause.head.arity))
        yield
    finally:
        for predicate, arity in current_clauses:
            args = ','.join(['_'] * arity)
            prolog.retractall(f'{predicate}({args})')


def pyswip_output_to_str(inp) -> Union[str, int, List]:
    if type(inp) is bytes:
        inp: bytes
        return f'"{inp.decode("utf-8")}"'
    elif type(inp) is int:
        inp: int
        return inp
    elif type(inp) is pyswip.easy.Variable:
        inp: pyswip.easy.Variable
        if inp.chars is None:
            return f"_{inp.handle}"
        else:
            return inp.chars
    elif type(inp) is pyswip.easy.Atom:
        inp: pyswip.easy.Atom
        return f"{inp.value}"
    elif type(inp) is list:
        inp: List
        return [pyswip_output_to_str(child) for child in inp]
    elif type(inp) is str:
        return inp
    else:
        raise ValueError(f"Unknwon type {type(inp)}")
