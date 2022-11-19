import { scene, getCurrentState } from "./main";
import * as THREE from "three";
import { performMoves } from "./keyListeners";
import { SetBackC, SetBackCC, SetDownC, SetDownCC, SetFrontC, SetFrontCC, SetLeftC, SetLeftCC, SetRightC, SetRightCC, SetUpC, SetUpCC } from "./arrows";
import { getAnimationTime } from "./cube";

var f_cw, r_cw, u_cw, b_cw, l_cw, d_cw;
var f_ccw, r_ccw, u_ccw, b_ccw, l_ccw, d_ccw;
var cw_buttons;
var ccw_buttons;

// var cw_moves = ["f", "r", "u", "b", "l", "d"];
// var ccw_moves = ["f'", "r'", "u'", "b'", "l'", "d'"];

export function setMoveButtons() {
  cw_buttons = document.getElementById("clockwise-buttons");
  ccw_buttons = document.getElementById("counter-clockwise-buttons");

  f_cw = document.getElementById("f_cw");
  r_cw = document.getElementById("r_cw");
  u_cw = document.getElementById("u_cw");
  b_cw = document.getElementById("b_cw");
  l_cw = document.getElementById("l_cw");
  d_cw = document.getElementById("d_cw");

  f_ccw = document.getElementById("f_ccw");
  r_ccw = document.getElementById("r_ccw");
  u_ccw = document.getElementById("u_ccw");
  b_ccw = document.getElementById("b_ccw");
  l_ccw = document.getElementById("l_ccw");
  d_ccw = document.getElementById("d_ccw");

  //Moves Pane Buttons
  mb_f_cw = document.getElementById("mb_f_cw")
  mb_r_cw = document.getElementById("mb_r_cw")
  mb_u_cw = document.getElementById("mb_u_cw")
  mb_b_cw = document.getElementById("mb_b_cw")
  mb_l_cw = document.getElementById("mb_l_cw")
  mb_d_cw = document.getElementById("mb_d_cw")

  mb_f_ccw = document.getElementById("mb_f_ccw")
  mb_r_ccw = document.getElementById("mb_r_ccw")
  mb_u_ccw = document.getElementById("mb_u_ccw")
  mb_b_ccw = document.getElementById("mb_b_ccw")
  mb_l_ccw = document.getElementById("mb_l_ccw")
  mb_d_ccw = document.getElementById("mb_d_ccw")

  setOnClickMoveFunctions();
  setHoverEffects();
}

function disableButtons() {
  for (var i = 0; i < cw_buttons.children.length; i++) {
    cw_buttons.children[i].disabled = true;
  }
  for (var i = 0; i < ccw_buttons.children.length; i++) {
    ccw_buttons.children[i].disabled = true;
  }
}

function enableButtons() {
  for (var i = 0; i < cw_buttons.children.length; i++) {
    cw_buttons.children[i].disabled = false;
  }
  for (var i = 0; i < ccw_buttons.children.length; i++) {
    ccw_buttons.children[i].disabled = false;
  }
}

function disableAllButtonsFor(ms = getAnimationTime()) {
  disableButtons();
  setTimeout(function () {
    enableButtons();
  }, ms);
}

function setOnClickMoveFunctions() {
  f_cw.onclick = function () {
    performMoves([], "f");
    disableAllButtonsFor();
  };
  r_cw.onclick = function () {
    performMoves([], "r");
    disableAllButtonsFor();
  };
  u_cw.onclick = function () {
    performMoves([], "u");
    disableAllButtonsFor();
  };
  b_cw.onclick = function () {
    performMoves([], "b");
    disableAllButtonsFor();
  };
  l_cw.onclick = function () {
    performMoves([], "l");
    disableAllButtonsFor();
  };
  d_cw.onclick = function () {
    performMoves([], "d");
    disableAllButtonsFor();
  };

  f_ccw.onclick = function () {
    performMoves([], "f'");
    disableAllButtonsFor();
  };
  r_ccw.onclick = function () {
    performMoves([], "r'");
    disableAllButtonsFor();
  };
  u_ccw.onclick = function () {
    performMoves([], "u'");
    disableAllButtonsFor();
  };
  b_ccw.onclick = function () {
    performMoves([], "b'");
    disableAllButtonsFor();
  };
  l_ccw.onclick = function () {
    performMoves([], "l'");
    disableAllButtonsFor();
  };
  d_ccw.onclick = function () {
    performMoves([], "d'");
    disableAllButtonsFor();
  };
}

function setHoverEffects() {
  f_cw.onmouseover = function () {
    SetFrontC(true);
  };
  f_cw.onmouseout = function () {
    SetFrontC(false);
  };
  r_cw.onmouseover = function () {
    SetRightC(true);
  };
  r_cw.onmouseout = function () {
    SetRightC(false);
  };
  u_cw.onmouseover = function () {
    SetUpC(true);
  };
  u_cw.onmouseout = function () {
    SetUpC(false);
  };
  b_cw.onmouseover = function () {
    SetBackC(true);
  };
  b_cw.onmouseout = function () {
    SetBackC(false);
  };
  l_cw.onmouseover = function () {
    SetLeftC(true);
  };
  l_cw.onmouseout = function () {
    SetLeftC(false);
  };
  d_cw.onmouseover = function () {
    SetDownC(true);
  };
  d_cw.onmouseout = function () {
    SetDownC(false);
  };

  f_ccw.onmouseover = function () {
    SetFrontCC(true);
  };
  f_ccw.onmouseout = function () {
    SetFrontCC(false);
  };
  r_ccw.onmouseover = function () {
    SetRightCC(true);
  };
  r_ccw.onmouseout = function () {
    SetRightCC(false);
  };
  u_ccw.onmouseover = function () {
    SetUpCC(true);
  };
  u_ccw.onmouseout = function () {
    SetUpCC(false);
  };
  b_ccw.onmouseover = function () {
    SetBackCC(true);
  };
  b_ccw.onmouseout = function () {
    SetBackCC(false);
  };
  l_ccw.onmouseover = function () {
    SetLeftCC(true);
  };
  l_ccw.onmouseout = function () {
    SetLeftCC(false);
  };
  d_ccw.onmouseover = function () {
    SetDownCC(true);
  };
  d_ccw.onmouseout = function () {
    SetDownCC(false);
    };

  //Moves Pane button hover effects
  mb_f_cw.onmouseover = function () {
      SetFrontC(true);
  };
  mb_f_cw.onmouseout = function () {
      SetFrontC(false);
  };
  mb_r_cw.onmouseover = function () {
      SetRightC(true);
  };
  mb_r_cw.onmouseout = function () {
      SetRightC(false);
  };
  mb_u_cw.onmouseover = function () {
      SetUpC(true);
  };
  mb_u_cw.onmouseout = function () {
      SetUpC(false);
  };
  mb_b_cw.onmouseover = function () {
      SetBackC(true);
  };
  mb_b_cw.onmouseout = function () {
      SetBackC(false);
  };
  mb_l_cw.onmouseover = function () {
      SetLeftC(true);
  };
  mb_l_cw.onmouseout = function () {
      SetLeftC(false);
  };
  mb_d_cw.onmouseover = function () {
      SetDownC(true);
  };
  mb_d_cw.onmouseout = function () {
      SetDownC(false);
  };

  mb_f_ccw.onmouseover = function () {
      SetFrontCC(true);
  };
  mb_f_ccw.onmouseout = function () {
      SetFrontCC(false);
  };
  mb_r_ccw.onmouseover = function () {
      SetRightCC(true);
  };
  mb_r_ccw.onmouseout = function () {
      SetRightCC(false);
  };
  mb_u_ccw.onmouseover = function () {
      SetUpCC(true);
  };
  mb_u_ccw.onmouseout = function () {
      SetUpCC(false);
  };
  mb_b_ccw.onmouseover = function () {
      SetBackCC(true);
  };
  mb_b_ccw.onmouseout = function () {
      SetBackCC(false);
  };
  mb_l_ccw.onmouseover = function () {
      SetLeftCC(true);
  };
  mb_l_ccw.onmouseout = function () {
      SetLeftCC(false);
  };
  mb_d_ccw.onmouseover = function () {
      SetDownCC(true);
  };
  mb_d_ccw.onmouseout = function () {
      SetDownCC(false);
  };
}
