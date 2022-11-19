from typing import Dict, List
import pytholog as pl


def main():
    kb = pl.KnowledgeBase("rubikscube")
    colors = ["w", "o", "r", "g", "b", "y"]

    # center cubelets
    kb.add_kn(["cubelet(X) :- center(X)"])
    for color in colors:
        kb.add_kn([f"center({color}_c)"])

        sticker_name: str = f"{color}_{color}_c"
        kb.add_kn([f"sticker({sticker_name})"])
        kb.add_kn([f"sticker_type({sticker_name},c)"])

    # opposite centers
    kb.add_kn(["opposite_face(X,Y) :- opposite_face_sym(X,Y),sticker(X),sticker(Y)"])
    kb.add_kn(["opposite_face(X,Y) :- opposite_face_sym(Y,X),sticker(X),sticker(Y)"])
    kb.add_kn(["opposite_face_sym(w_w_c,y_y_c)"])
    kb.add_kn(["opposite_face_sym(r_r_c,o_o_c)"])
    kb.add_kn(["opposite_face_sym(b_b_c,g_g_c)"])

    # clockwise
    clockwise: Dict[str, List[str]] = dict()
    clockwise["w_w_c"] = ["b_b_c", "r_r_c", "g_g_c", "o_o_c"]
    clockwise["r_r_c"] = ["y_y_c", "g_g_c", "w_w_c", "b_b_c"]
    clockwise["b_b_c"] = ["r_r_c", "w_w_c", "o_o_c", "y_y_c"]

    for center in clockwise.keys():
        for idx in range(len(clockwise[center])):
            c1 = clockwise[center][idx]
            c2 = clockwise[center][(idx + 1) % len(clockwise[center])]
            kb.add_kn([f"clockwise({center},{c1},{c2})"])

            center_opps = kb.query(pl.Expr(f"opposite_face({center},X)"))
            assert len(center_opps) == 1
            kb.add_kn([f"clockwise({center_opps[0]['X']},{c2},{c1})"])

    # counter clockwise
    kb.add_kn(["counterclockwise(X,Y,Z):-clockwise(X,Z,Y),center(X),center(Y),center(Z)"])

    # adjacent centers
    kb.add_kn(["adjacent(X,Y):-clockwise(X,Y,Z)"])

    # edge cubelets
    kb.add_kn(["cubelet(X):-edge(X)"])
    for center_adj in kb.query(pl.Expr("adjacent(w_w_c,X)")):
        kb.add_kn([f"edge(w{center_adj['X'][0]}_e)"])
    for center_adj in kb.query(pl.Expr("adjacent(y_y_c,X)")):
        kb.add_kn([f"edge(y{center_adj['X'][0]}_e)"])
    kb.add_kn(["edge(br_e)", "edge(bo_e)", "edge(go_e)", "edge(gr_e)"])

    import pdb
    pdb.set_trace()

    kb.add_kn(["same_face(X,Y) :- same_face_sym(X,Y); same_face_sym(Y,X)"])
    kb.add_kn(["not(same_face(X,Y)) :- not(same_face_sym(X,Y)); not(same_face_sym(Y,X))"])
    for edge_q in kb.query(pl.Expr("edge(X)")):
        edge = edge_q['X']
        sticker1_name: str = f"{edge[0]}_{edge}"
        sticker2_name: str = f"{edge[1]}_{edge}"

        kb.add_kn([f"sticker({sticker1_name})", f"sticker({sticker2_name})"])
        kb.add_kn([f"sticker_type({sticker1_name},e)", f"sticker_type({sticker2_name},e)"])

        kb.add_kn([f"not(same_face_sym({sticker2_name},Y)) :- same_face_sym({sticker1_name},X),opposite_face(X,Y)"])
        kb.add_kn([f"not(same_face_sym({sticker1_name},Y)) :- same_face_sym({sticker2_name},X),opposite_face(X,Y)"])

    kb.add_kn([f"not(same_face_sym(X,Z)) :- same_face_sym(X,Y),opposite_face(Y,Z)"])
    kb.add_kn([f"not(same_face_sym(X,Z)) :- same_face_sym(X,Y),adjacent(Y,Z)"])

    # solved
    kb.add_kn(["solved :- same_face_sym(w_wo_e,w_w_c),same_face_sym(o_wo_e,o_o_c)"])

    kb.add_kn(["same_face_sym(w_wo_e,b_b_c)"])
    kb.add_kn(["same_face_sym(o_wo_e,o_o_c)"])

    print(list(kb.query(pl.Expr("solved"))))
    print(list(kb.query(pl.Expr("not(same_face(w_wo_e,Y))"))))

    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    main()
