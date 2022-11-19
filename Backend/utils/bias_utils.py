from typing import List, Optional


class BiasPredicate:
    def __init__(self, name: str, types: List[str], directions: List[str], loc: str):
        self.name: str = name
        self.types: List[str] = types
        self.directions: List[str] = directions
        self.arity: int = len(self.directions)
        self.loc: str = loc


def type_limit(var_type: str, num_times: int) -> str:
    bias_str: str = ":- #count{Var : var_type(Clause,Var,%s)} > %i." % (var_type, num_times)
    return bias_str


def pred_limit(pred: BiasPredicate, num_times: int) -> str:
    bias_str: str = ":- #count{Vars: body_literal(Clause, %s, %i, Vars)} > %i." % (pred.name, pred.arity, num_times)
    return bias_str


def body_head(pred: BiasPredicate) -> str:
    loc: str = pred.loc
    if loc == "head":
        body_head_pre: str = "head_pred"
    elif loc == "body":
        body_head_pre: str = "body_pred"
    else:
        raise ValueError(f"Unknown body/head location {loc}")

    body_head_str: str = f"{body_head_pre}({pred.name}, {pred.arity})."

    return body_head_str


def pred_type(pred: BiasPredicate) -> Optional[str]:
    if len(pred.types) == 0:
        pred_type_str = None
    else:
        type_str: str = ", ".join(pred.types)
        if len(pred.types) == 1:
            type_str = f"{type_str},"

        pred_type_str = f"type({pred.name}, ({type_str}))."

    return pred_type_str


def pred_direction(pred: BiasPredicate) -> str:
    direc_str: str = ", ".join(pred.directions)
    if len(pred.directions) == 1:
        direc_str = f"{direc_str},"

    pred_direc_str: str = f"direction({pred.name}, ({direc_str}))."
    return pred_direc_str
