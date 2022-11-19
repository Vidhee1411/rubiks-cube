from popper.aspsolver import Clingo
from torch.multiprocessing import get_context
from popper.generate import generate_program


def _clingo_proc(args_queue, return_queue, bias_file_name: str, clingo_args):
    solver = Clingo(bias_file_name, clingo_args)
    while True:
        args = args_queue.get()
        if args is None:
            break
        elif args[0] == "max_clauses":
            return_queue.put(solver.max_clauses)
        elif args[0] == "max_vars":
            return_queue.put(solver.max_vars)
        elif args[0] == "update_number_of_literals":
            solver.update_number_of_literals(args[1])
        elif args[0] == "add_ground_clauses":
            solver.add_ground_clauses(args[1])
        elif args[0] == "ground_program":
            ret = solver.ground_program(args[1], args[2], args[3])
            return_queue.put(ret)
        elif args[0] == "get_model":
            ret = solver.get_model()
            if ret is not None:
                ret = generate_program(ret)
            return_queue.put(ret)
        else:
            raise ValueError("Unknown clingo method %s" % args[0])


class ClingoProc:
    def __init__(self, bias_file_name: str, clingo_args):
        ctx = get_context("spawn")
        self.args_queue: ctx.Queue = ctx.Queue()
        self.return_queue: ctx.Queue = ctx.Queue()

        self.proc = ctx.Process(target=_clingo_proc,
                                args=(self.args_queue, self.return_queue, bias_file_name, clingo_args))
        self.proc.daemon = True
        self.proc.start()

        self.args_queue.put(("max_clauses",))
        self.max_clauses = self.return_queue.get()

        self.args_queue.put(("max_vars",))
        self.max_vars = self.return_queue.get()

    def update_number_of_literals(self, size):
        self.args_queue.put(("update_number_of_literals", size))

    def add_ground_clauses(self, clauses):
        self.args_queue.put(("add_ground_clauses", clauses))

    def ground_program(self, constraint, max_clauses, max_vars):
        self.args_queue.put(("ground_program", constraint, max_clauses, max_vars))
        return self.return_queue.get()

    def get_model(self):
        self.args_queue.put(("get_model",))
        return self.return_queue.get()

    def close(self):
        self.args_queue.put(None)
        self.proc.join()
