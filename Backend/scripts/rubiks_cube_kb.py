from typing import Dict, List
from pyswip import Prolog


def lquery(prolog: Prolog, query: str):
    return list(prolog.query(query))


def main():
    # C: color, P: Piece (cubelet), F: face, S: Sticker
    prolog = Prolog()
    colors = ["w", "o", "r", "g", "b", "y"]

    # center cubelets and faces
    prolog.assertz("piece(X):-center(X)")
    for color in colors:
        prolog.assertz(f"center({color}_c)")

        face_name: str = f"{color}_f"
        prolog.assertz(f"face({face_name})")
        prolog.assertz(f"color({face_name}, {color})")

    # opposite centers
    prolog.assertz("opposite_face(X,Y) :- opposite_face_sym(X,Y),face(X),face(Y)")
    prolog.assertz("opposite_face(X,Y) :- opposite_face_sym(Y,X),face(X),face(Y)")
    prolog.assertz("opposite_face_sym(w_f, y_f)")
    prolog.assertz("opposite_face_sym(r_f, o_f)")
    prolog.assertz("opposite_face_sym(b_f, g_f)")

    # clockwise
    clockwise: Dict[str, List[str]] = dict()
    clockwise["w_f"] = ["b_f", "r_f", "g_f", "o_f"]
    clockwise["r_f"] = ["y_f", "g_f", "w_f", "b_f"]
    clockwise["b_f"] = ["r_f", "w_f", "o_f", "y_f"]

    for center in clockwise.keys():
        for idx in range(len(clockwise[center])):
            c1 = clockwise[center][idx]
            c2 = clockwise[center][(idx + 1) % len(clockwise[center])]
            prolog.assertz(f"clockwise({center},{c1},{c2})")

            center_opps = list(prolog.query(f"opposite_face({center},X)"))
            assert len(center_opps) == 1
            prolog.assertz(f"clockwise({center_opps[0]['X']},{c2},{c1})")

    # counter clockwise
    prolog.assertz("counterclockwise(X,Y,Z):-clockwise(X,Z,Y),center(X),center(Y),center(Z)")

    # adjacent centers
    prolog.assertz("adjacent(X,Y):-clockwise(X,Y,Z)")

    # edge cubelets
    prolog.assertz("piece(X):-edge(X)")
    for center_adj in list(prolog.query("adjacent(w_f,X)")):
        prolog.assertz(f"edge(w{center_adj['X'][0]}_e)")
    for center_adj in list(prolog.query("adjacent(y_f,X)")):
        prolog.assertz(f"edge(y{center_adj['X'][0]}_e)")
    prolog.assertz("edge(br_e)")
    prolog.assertz("edge(bo_e)")
    prolog.assertz("edge(go_e)")
    prolog.assertz("edge(gr_e)")

    prolog.assertz("in_place(P) :- center(P)")
    prolog.assertz("in_place(P) :- cpf(C1, P, F1), cpf(C2, P, F2), color(F1, F1_col), color(F2, F2_col), "
                   "F1_col == C1, F2_col == C2, F1 \== F2, edge(P)")

    prolog.assertz("cpf(w, wg_e, w_f)")
    prolog.assertz("cpf(g, wg_e, g_f)")
    prolog.assertz("cpf(w, wo_e, w_f)")
    prolog.assertz("cpf(o, wo_e, o_f)")

    print(list(prolog.query("in_place(w_c)")))
    print(list(prolog.query("in_place(wg_e)")))
    print(list(prolog.query("in_place(wo_e)")))

    import pdb
    pdb.set_trace()

    # solved
    print(list(prolog.query("on_piece(S1,P1), on_face(S1,F1), color(S1) \== color(F1), "
                            "on_piece(S2,P1), on_face(S2, F2), color(S2) \== color(F2), "
                            "in_place(P2), has_color(color(S1), P2), has_color(color(F1), P2), "
                            "in_place(P3), has_color(color(S1), P3), has_color(color(F2), P3), "
                            "edge(P1), edge(P2), edge(P3), S1 \== S2, F1 \== F2, P1 \== P2, P1 \== P3, P2 \== P3")))

    query = "cpf(P1_col1, P1, F1), color(F1, F1_col), P1_col1 \== F1_col, " \
            "cpf(P1_col2, P1, F2), color(F2, F2_col), P1_col2 \== F2_col, " \
            "in_place(P2), cpf(P1_col1, P2, F3), cpf(F1_col, P2, F1)" \
            "in_place(P3), cpf(P1_col1, P3, F3), cpf(F2_col, P3, F2)" \
            "edge(P1), edge(P2), edge(P3), P1_col1 \== P1_col2, F1 \== F2, F1 \== F3, F2 \== F3, " \
            "P1 \== P2, P1 \== P3, P2 \== P3"

    print(list(prolog.query("is_solved")))
    print(list(prolog.query("not(same_face(w_wo_e,Y))")))


if __name__ == "__main__":
    main()
