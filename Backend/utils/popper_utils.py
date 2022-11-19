from typing import Tuple, List, Optional
from popper.log import Experiment
from popper.tester import Tester
from utils.clingo_utils import ClingoProc
from popper.constrain import Constrain, Outcome
from popper.core import Program
from popper.core import Clause, Literal
import re
import time


def get_head_body_code(prog: Program) -> Tuple[str, str]:
    prog_code: str = list(prog.to_code())[0].replace(".", "")

    head_body: List[str] = prog_code.split(":-")
    head_body = [x.strip() for x in head_body]

    return head_body[0], head_body[1]


def query_examples(tester: Tester, prog: Program, ex_type: str) -> List[str]:
    with tester.using(prog):
        results = [x['X'] for x in list(tester.prolog.query(f'{ex_type}(X), test_ex(X)'))]

    return results


def get_all_matching_body(tester: Tester, body_code: str):
    with tester.using_clause_str(f'move(A,Dummy2,Dummy3):- {body_code}',
                                 'move(_,_,_)'):
        results = [x['X'] for x in list(tester.prolog.query('pos(X), test_ex(X)'))]

    return results


def parse_exs(exs_file_name: str) -> Tuple[List[str], List[str]]:
    exs_file = open(exs_file_name, "r")
    pos_examples: List[str] = []
    neg_examples: List[str] = []
    for line in exs_file:
        regex = re.search("(pos|neg)\((.*)\)", line)
        if regex:
            example: str = regex.group(2)
            if regex.group(1) == "pos":
                pos_examples.append(example)
            elif regex.group(1) == "neg":
                neg_examples.append(example)
            else:
                raise ValueError(f"Unknown pos/neg type {regex.group(1)}")

    exs_file.close()

    return pos_examples, neg_examples


def ground_constraints(grounder, max_clauses, max_vars, constraints):
    for constraint in constraints:
        # find bindings for variables in the constraint
        assignments = grounder.ground_program(constraint, max_clauses, max_vars)

        # build the clause
        clause = Clause(constraint.head, tuple(lit for lit in constraint.body if isinstance(lit, Literal)))

        # ground the clause for each variable assignment
        for assignment in assignments:
            yield clause.ground(assignment)


def pprint(program):
    for clause in program.to_code():
        print(clause)


def popper(bias_file_name: str, tester: Tester, max_literals: int, best_tp: int = 0, partial: bool = False,
           debug: bool = False, stats: bool = False, time_lim: Optional[float] = None) -> Tuple[Optional[Program], int]:
    spec_possible: List[Tuple[Program, int]] = []
    if partial:
        assert tester.test_all_pos, "Must test all positive if finding partial solutions"

    start_time = time.time()
    if time_lim is None:
        time_lim = float('inf')

    experiment = Experiment(debug=debug, stats=stats)
    solver = ClingoProc(bias_file_name, experiment.clingo_args)
    constrainer = Constrain()
    num_solutions = 0

    best_program: Optional[Program] = None

    # TODO, multiply with max clauses?
    for size in range(1, max_literals + 1):
        if experiment.debug:
            print(f'{"*" * 20} MAX LITERALS: {size} {"*" * 20}')
        solver.update_number_of_literals(size)
        while True:
            # 1. Generate
            with experiment.duration('generate'):
                program: Optional[Program] = solver.get_model()
                if not program:
                    break

            experiment.total_programs += 1

            # 2. Test
            with experiment.duration('test'):
                (outcome, (TP, FN, TN, FP)) = tester.test(program)

            if experiment.debug:
                print(f'Program {experiment.total_programs}:')
                pprint(program)
                approx_pos = '+' if TP + FN < tester.num_pos else ''
                approx_neg = '+' if TN + FP < tester.num_neg else ''
                print(f'TP: {TP}{approx_pos}, FN: {FN}{approx_pos}, TN: {TN}{approx_neg}, FP: {FP}{approx_neg}')

            if outcome == (Outcome.ALL, Outcome.NONE):
                if experiment.debug:
                    print()
                if experiment.stats:
                    experiment.show_stats()

                if experiment.debug:
                    print('SOLUTION:')
                    pprint(program)

                num_solutions += 1
                if num_solutions == experiment.max_solutions:
                    solver.close()
                    return program, TP

            # 3. Build constraints
            cons = set()

            if experiment.functional_test and tester.is_non_functional(program):
                cons.update(constrainer.generalisation_constraint(program))

            # eliminate generalisations of clauses that contain redundant literals
            # for clause in tester.check_redundant_literal(program):
            #    cons.update(constrainer.redundant_literal_constraint(clause))

            # eliminate generalisations of programs that contain redundant clauses
            # if tester.check_redundant_clause(program):
            #    cons.update(constrainer.generalisation_constraint(program))

            # add other constraints
            if partial:
                cons.update(constrainer.banish_constraint(program))

                if FP == 0:
                    cons.update(constrainer.specialisation_constraint(program))
                    if TP > best_tp:
                        best_program = program
                        best_tp = TP

                        # update possible specialisations
                        spec_constr = [prog_i for prog_i, TP_i in spec_possible if TP_i <= best_tp]
                        for prog_i in spec_constr:
                            cons.update(constrainer.specialisation_constraint(prog_i))

                        spec_possible = [(prog_i, TP_i) for prog_i, TP_i in spec_possible if TP_i > best_tp]
                else:
                    cons.update(constrainer.generalisation_constraint(program))
                    if TP > best_tp:
                        spec_possible.append((program, TP))
                    else:
                        cons.update(constrainer.specialisation_constraint(program))

                if experiment.debug and (best_program is not None):
                    print(f"Best program ({best_tp}): ", end="")
                    pprint(best_program)
            else:
                cons.update(constrainer.build_constraints(program, outcome))

            if experiment.debug:
                print('Constraints:')
                for constraint in cons:
                    print(constraint.ctype, constraint)

            # 4. Ground constraints
            with experiment.duration('ground'):
                cons_g = set(ground_constraints(solver, solver.max_clauses, solver.max_vars, cons))

            # 5. Add to the solver
            with experiment.duration('add'):
                solver.add_ground_clauses(cons_g)

            if experiment.debug:
                print("")

            if (time.time() - start_time) > time_lim:
                solver.close()
                return None,  0

    if experiment.stats:
        experiment.show_stats()

    if experiment.debug:
        print('NO MORE SOLUTIONS')

    solver.close()

    return best_program, best_tp


def literal_args_asp(literal: Literal) -> str:
    num_args: int = len(literal.arguments)
    arg_names: str = ",".join(literal.arguments)
    if num_args == 1:
        arg_names = f"{arg_names},"
    asp_args: str = f"{literal.predicate}, {num_args}, ({arg_names})"

    return asp_args
