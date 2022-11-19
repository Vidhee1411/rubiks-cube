max_clauses(1).
max_vars(4).
max_body(8).

% head and body
head_pred(precond, 1).
body_pred(cubelet, 1).
body_pred(has_stk, 2).
body_pred(match_0, 3).
body_pred(match_0_flip, 3).
body_pred(match_90, 3).
body_pred(match_180, 3).
body_pred(match_180_flip, 3).
body_pred(white, 1).
body_pred(yellow, 1).

% types
type(precond, (state,)).
type(cubelet, (cubelet,)).
type(has_stk, (cubelet, sticker)).
type(match_0, (state, cubelet, sticker)).
type(match_0_flip, (state, cubelet, sticker)).
type(match_90, (state, cubelet, sticker)).
type(match_180, (state, cubelet, sticker)).
type(match_180_flip, (state, cubelet, sticker)).

% directions
direction(precond, (in,)).
direction(cubelet, (out,)).
direction(has_stk, (in, out)).
direction(match_0, (in, in, in)).
direction(match_0_flip, (in, in, in)).
direction(match_90, (in, in, in)).
direction(match_180, (in, in, in)).
direction(match_180_flip, (in, in, in)).
direction(white, (in,)).
direction(yellow, (in,)).

% other constraints

% Variable helpers
head_var(Clause, Var) :- head_literal(Clause, _, _, Vars), var_member(Var, Vars).
body_var(Clause, Var) :- body_literal(Clause, _, _, Vars), var_member(Var, Vars).
body_pred_vars(Clause, Pred, Vars) :- body_literal(Clause, Pred, _, Vars).
head_pred_var(Clause, Pred, Var) :- head_literal(Clause, Pred, _, Vars), var_member(Var, Vars).
body_pred_var(Clause, Pred, Var) :- body_literal(Clause, Pred, _, Vars), var_member(Var, Vars).

% Predicate types
color_pred(white).
color_pred(yellow).
color_pred(orange).
color_pred(red).
color_pred(blue).
color_pred(green).
match_pred(match_0).
match_pred(match_0_flip).
match_pred(match_180).
match_pred(match_180_flip).
match_pred(match_cl).
match_pred(match_cc).
match_pred(match_90).

%%% Cubelet constraints

%%% Sticker constraints

% Restrict number of variables (max two stk needed per cubelet)
:- body_literal(C, cubelet, 1, (Cbl,)), #count{Stk: body_literal(C, has_stk, 2, (Cbl, Stk))} > 2.

% More than one sticker on the same cubelet cannot be the same color
:- clause(C), color_pred(P), body_literal(C, cubelet, 1, (Cbl,)), #count{Stk: body_literal(C, P, 1, (Stk,)), body_literal(C, has_stk, 2, (Cbl, Stk))} > 1.

 % Sticker can only be in one match_pred
:- clause(C), head_var(C, A), body_literal(C, has_stk, 2, (Cbl, Stk)), #count{P: body_literal(C, P, 3, (A, Cbl, Stk)), match_pred(P)} > 1.

%%% Color constraints

% Only face and sticker in color pred
:- body_literal(C, P, 1, (Obj,)), color_pred(P), var_type(C, Obj, Type), Type != face, Type != sticker.

% Object cannot be more than one color type
:- body_var(C, Obj), #count{P: body_literal(C, P, 1, (Obj,)), color_pred(P)} > 1.
