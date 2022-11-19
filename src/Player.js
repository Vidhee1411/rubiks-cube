import {resetScene, getAnimationTime, resetCubeConfiguration} from "./cube";
import { getCurrentMoves, performMoves } from "./keyListeners";
import { getCurrentState, resetCameraPosition, pageNotInitialized } from "./main";
import {applyState, applyStateIfInitial, updateState} from "./state";
import {getDatabase, onValue, ref} from "firebase/database";
import {app} from "./firebaseCred";
import {getAuth} from "firebase/auth";
import { pageInitialized } from "./UIGuide/guide";
// import arrowImage from "./images/arrow.png";

export var currentState = [];
export var currentMoves = [];

export var currentIndex = null;

var moveUIList;

export function setOnClickFunctions() {
  // var arrow = document.getElementById("forwardButtonImage");
  // arrow.src = arrowImage;

  //REWIND BUTTON
  document.getElementById("rewindButton").onclick = function () {
    disableButton("rewindButton");
    setTimeout(() => {
      if (currentIndex > 0) enableButton("rewindButton");
    }, getAnimationTime() * 1.5);
    if (currentIndex == null) {
      console.log("No actions in system");
    } else if (currentMoves.length != 0 && currentIndex > 0) {
      performMoves([], getOpposite(currentMoves[currentIndex - 1]));
      currentIndex--;
      enableButton("forwardButton");
    }
    if (currentIndex == 0) {
      disableButton("rewindButton");
    }
    console.log("Clicked on rewind Button", currentIndex);
  };

  //FORWARD BUTTON
  document.getElementById("forwardButton").onclick = function () {
    disableButton("forwardButton");
    setTimeout(() => {
      if (currentIndex < currentMoves.length) enableButton("forwardButton");
    }, getAnimationTime() * 1.5);
    if (currentIndex == null) {
      console.log("No actions in system");
    } else if (currentMoves.length != 0 && currentIndex < currentMoves.length) {
      performMoves([], currentMoves[currentIndex]);
      currentIndex++;
      if (currentIndex > 0) enableButton("rewindButton");
    }

    if (currentIndex == currentMoves.length) {
      disableButton("forwardButton");
    }
    console.log("Clicked on forward Button", currentIndex);
  };

  //RESET BUTTON
  document.getElementById("resetButton").onclick = function () {
    resetCameraPosition();
  };

  //RESET CUBE BUTTON
    document.getElementById("resetCubeButton").onclick = function () {
        if (pageNotInitialized && pageInitialized) {
          resetCameraPosition();
          applyState(getCurrentState());
      }
      else {
          resetCubeConfiguration();
      }

        updateState('', true);
        console.log("Clicked on reset cube Button", currentIndex);
  }

  //MoveUI List
    moveUIList = document.getElementById("movesList");

/*
  //CUSTOMIZE CUBE BUTTON
  document.getElementById("resetCubeButton").onclick = function () {
    if (pageInitialized != true) {
      resetCameraPosition();
      applyState(getCurrentState());
    }
    else
      resetCubeConfiguration();

    console.log("Clicked on reset cube Button", currentIndex);
  }
*/
}

export function getOpposite(move) {
  if (move[1] == "'") return move[0];
  else return move[0] + "'";
}

export function setCurrentState(state) {
  currentState = state;
}
export function setCurrentMoves(moves) {
  currentMoves = moves;
  currentIndex = moves.length;
  moveUIList.innerText = "";
  for (var i = 0; i < currentMoves.length; i++) {
    var li = document.createElement("li");
    li.innerText = currentMoves[i];
    moveUIList.append(li);
  }
}

export function enableButton(id) {
  if (currentIndex > 0) document.getElementById(id).disabled = false;
  if (id == "forwardButton") document.getElementById(id).disabled = false;
}
export function disableButton(id) {
  document.getElementById(id).disabled = true;
}
