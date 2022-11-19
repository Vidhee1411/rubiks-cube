#!/usr/bin/env python3

from typing import List
import argparse

from popper.log import Experiment
from popper.aspsolver import Clingo
from popper.tester import Tester
from popper.constrain import Constrain
from popper.generate import generate_program
from popper.core import Clause, Literal
from utils import popper_utils
import os
import time

from collections import OrderedDict


def parse_args():
    parser = argparse.ArgumentParser(description='Popper, an ILP engine based on learning from failures')
    parser.add_argument('kbpath', help='Path to the knowledge base one wants to learn on')
    parser.add_argument('--eval-timeout', type=float, default=0.1, help='Prolog evaluation timeout in seconds')
    parser.add_argument('--timeout', type=float, default=600, help='Overall timeout (in seconds)')
    parser.add_argument('--max-literals', type=int, default=100, help='Maximum number of literals allowed in program')
    parser.add_argument('--max-solutions', type=int, default=1, help='Maximum number of solutions to print')
    parser.add_argument('--debug', default=False, action='store_true', help='Print debugging information to stderr')
    parser.add_argument('--stats', default=False, action='store_true', help='Print statistics at end of execution')
    parser.add_argument('--functional-test', default=False, action='store_true', help='Run custom functional test')
    parser.add_argument('--clingo-args', type=str, default='', help='Arguments to pass to Clingo')

    return parser.parse_args()


def ground_constraints(grounder, max_clauses, max_vars, constraints):
    for constraint in constraints:
        # find bindings for variables in the constraint
        assignments = grounder.ground_program(constraint, max_clauses, max_vars)

        # build the clause
        clause = Clause(constraint.head, tuple(lit for lit in constraint.body if isinstance(lit, Literal)))

        # ground the clause for each variable assignment
        for assignment in assignments:
            yield clause.ground(assignment)


def apply_constraint(prog, experiment, solver, constrainer, const_types: List[str]):
    # get all constraints
    cons = set()
    for const_type in const_types:
        if const_type == 'banish':
            cons.update(constrainer.banish_constraint(prog))
        elif const_type == 'generalisation':
            cons.update(constrainer.generalisation_constraint(prog))
        elif const_type == 'specialisation':
            cons.update(constrainer.specialisation_constraint(prog))
        elif const_type == 'redundancy':
            cons.update(constrainer.redundancy_constraint(prog))
        elif const_type == 'redundant_literal':
            cons.update(constrainer.redundant_literal_constraint(prog))
        else:
            raise ValueError("Unknown constraint type %s" % const_type)

    # apply constraints
    with experiment.duration('ground'):
        cons = set(ground_constraints(solver, solver.max_clauses, solver.max_vars, cons))

    with experiment.duration('add'):
        solver.add_ground_clauses(cons)


def pprint(program):
    for clause in program.to_code():
        print(clause)


def get_highest_matching_clause(experiment, examples: List, ancestor_progs: List):
    if experiment.kbpath[-1] != '/':
        experiment.kbpath += '/'

    solver = Clingo(experiment.kbpath, experiment.clingo_args)
    tester = Tester(experiment.kbpath, examples, [], test_all=True)
    constrainer = Constrain()

    import pdb
    pdb.set_trace()

    # remove ancestor programs and their generalizations
    for prog in ancestor_progs:
        apply_constraint(prog, experiment, solver, constrainer, ["banish", "generalisation"])

    times = OrderedDict()
    times["generate"] = 0.0
    times["test"] = 0.0
    times["constrain"] = 0.0

    best_program = None
    best_subset_size: int = 0
    for size in range(1, experiment.args.max_literals + 1):
        # if experiment.debug:
        #    print(f'{"*" * 20} MAX LITERALS: {size} {"*" * 20}')
        solver.update_number_of_literals(size)
        while True:
            # 1. Generate
            start_time = time.time()
            with experiment.duration('generate'):
                model = solver.get_model()
                if not model:
                    break
                program = generate_program(model)

            experiment.total_programs += 1

            times["generate"] += time.time() - start_time

            # 2. Test
            start_time = time.time()
            with experiment.duration('test'):
                (outcome, (TP, FN, TN, FP)) = tester.test(program)

            pos: int = TP + FP
            neg: int = TN + FN
            total_exs: int = pos + neg
            subset_size: int = pos
            if (subset_size > best_subset_size) and (subset_size < total_exs):
                best_program = program
                best_subset_size = subset_size

            if experiment.debug:
                print(f'Program {experiment.total_programs} ({subset_size}): ', end="")
                pprint(program)
                # print(f"Subset size: {subset_size}, Positive: {pos}, Negative: {neg}")

                if best_program is not None:
                    print(f"Best program ({best_subset_size}): ", end="")
                    pprint(best_program)
            times["test"] += time.time() - start_time

            # if experiment.debug:
            #    print('Constraints:')
            #    for constraint in cons:
            #        print(constraint.ctype, constraint)
            #    print()

            start_time = time.time()
            apply_constraint(program, experiment, solver, constrainer, ["banish"])
            if (best_program is not None) and (subset_size <= best_subset_size):
                # disallow all specializations
                apply_constraint(program, experiment, solver, constrainer, ["specialisation"])
            times["constrain"] += time.time() - start_time

            if experiment.debug:
                time_str_elems = []
                for key, val in times.items():
                    time_str_elems.append("%s: %.2f" % (key, val))

                time_str: str = ", ".join(time_str_elems)

                print(f"Times - {time_str}")

            print("")

    if experiment.stats:
        experiment.show_stats()

    print("Search Done\n")
    exs_prog, exs_body, exs_remaining = print_stats(best_program, tester)

    tester.close()

    return best_program, exs_prog, exs_body, exs_remaining


def print_stats(program, tester: Tester):
    print("Best program")
    pprint(program)
    num_tot: int = tester.num_pos + tester.num_neg
    exs_prog = popper_utils.query_examples(tester, program)
    exs_body = popper_utils.get_all_matching_body(tester, popper_utils.get_head_body_code(program)[1])

    print(f"{num_tot} total examples")
    print(f"{len(exs_prog)} out of {len(exs_body)} examples that match program body execute program.")

    exs_all = [x['X'] for x in list(tester.prolog.query('pos(X)'))]
    exs_remaining = list(set(exs_all) - set(exs_body))

    print(f"{len(exs_remaining)} examples remaining after applying program")

    return exs_prog, exs_body, exs_remaining


def main():
    # parse arguments
    args = parse_args()
    ancestor_progs: List = []

    # get examples
    exs_file_name: str = os.path.join(args.kbpath, "exs.pl")
    pos_examples, neg_examples = popper_utils.parse_exs(exs_file_name)
    assert len(neg_examples) == 0, "There should be no negative examples"
    print(f"{len(pos_examples)} positive examples and {len(neg_examples)} negative examples.")

    # get best probram
    experiment = Experiment(args.kbpath, debug=args.debug, stats=args.stats, max_solutions=args.max_solutions,
                            functional_test=args.functional_test, clingo_args=args.clingo_args)
    best_prog, exs_prog, exs_body, exs_remaining = get_highest_matching_clause(experiment, pos_examples, ancestor_progs)
    ancestor_progs.append(best_prog)
    print("")

    get_highest_matching_clause(experiment, exs_body, ancestor_progs)


if __name__ == '__main__':
    main()
