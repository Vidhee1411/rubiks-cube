from typing import List
from popper.core import Program
from popper.tester import Tester


def get_matches(kbpath: str, prog: Program, exs: List[str]) -> List[str]:
    tester = Tester(kbpath, exs, [])

    with tester.using(prog):
        results = [x['X'] for x in list(tester.prolog.query('pos(X), test_ex(X)'))]

    return results


def print_program(program):
    for clause in program.to_code():
        print(clause)

