import { turnX, turnY, turnZ, resetScene, getAnimationTime, enableSpeedSlider } from "./cube";
import { applyState, updateState } from "./state";
import { scene, state } from "./main";
import { disableButton, enableButton } from "./Player";
import * as THREE from "three";
import { SetFrontC, SetFrontCC, SetBackC, SetBackCC, SetDownC, SetDownCC, SetLeftC, SetLeftCC, SetRightC, SetRightCC, SetUpC, SetUpCC } from "./arrows";
import { app } from "./firebaseCred";
import { getDatabase, ref, onValue, set, update } from "firebase/database";
// Moves to be made
export var moves = "ff'b'rrrl'uduudd'";
export var currentMoves = [];

const database = getDatabase(app);

export function onDocumentKeyDown(event) {
  var keyCode = event.which;
  if (keyCode == 90) {
    //Z cw
    turnZ(getRandomMove(), getRandomMove());
  } else if (keyCode == 67) {
    //C cw
    turnY(getRandomMove(), getRandomMove());
  } else if (keyCode == 88) {
    //X cw
    turnX(getRandomMove(), getRandomMove());
  } else if (keyCode == 70) {
    // Turn Front face clockwise - F
    turnZ(1, 0);
  } else if (keyCode == 71) {
    // Turn Front face counter clockwise - G
    turnZ(1, 1);
  } else if (keyCode == 66) {
    // Turn Back face clockwise - B
    turnZ(-1, 1);
  } else if (keyCode == 78) {
    // Turn Back face counter clockwise - N
    turnZ(-1, 0);
  } else if (keyCode == 76) {
    // Turn Left face clockwise - L
    turnX(-1, 1);
  } else if (keyCode == 75) {
    // Turn Left face counter clockwise - K
    turnX(-1, 0);
  } else if (keyCode == 82) {
    // Turn Right face clockwise - R
    turnX(1, 0);
  } else if (keyCode == 84) {
    // Turn Right face counter lockwise - T
    turnX(1, 1);
  } else if (keyCode == 85) {
    // Turn Up face clockwise - U
    turnY(1, 1);
  } else if (keyCode == 73) {
    // Turn Up face counter clockwise - I
    turnY(1, 0);
  } else if (keyCode == 68) {
    // Turn Down face clockwise - D
    turnY(-1, 0);
  } else if (keyCode == 83) {
    // Turn Down face counter clockwise - S
    turnY(-1, 1);
  } else if (keyCode == 16) {
    performMoves(moves_array);
  } else if (keyCode == 13) {
    //Gets state of the rubiks cube from python server and applies it to cubeArray
    $(function () {
      $.ajax({
        url: "http://127.0.0.1:5000/initState",
        data: {},
        type: "POST",
        dataType: "json",
        success: function (response) {
          //console.log("success\nresponse:", response);
          state = response["state"];
          state = JSON.parse(state);
          //console.log("state:", state);
          resetScene();
          applyState(state);
        },
        error: function (error) {
          //console.log("Holy erros Batman!");
        },
      });
      //console.log("ready!");
    });

    //This can be used to connect to chatbot for message transfers
    // $(function () {
    //   $.ajax({
    //     url: "http://localhost:5005/webhooks/rest/webhook",
    //     data: JSON.stringify({ sender: "someone", message: "hi" }),
    //     type: "POST",
    //     dataType: "json",
    //     success: function (response) {
    //       console.log("success\nresponse:", response);
    //     },
    //     error: function (error) {
    //       console.log("Holy erros Batman!");
    //     },
    //   });
    //   console.log("ready!");
    // });
  }
}

export function getCurrentMoves() {
  return currentMoves;
}

export async function performMoves(moves_array, moves) {
  // Create an array of the moves - attach an apostrophe to the move prior
  //
  for (let j = 0; j < moves.length; j++) {
    if (moves[j + 1] == "'") {
      moves_array.push(moves[j] + moves[j + 1]);
    } else if (moves[j] == "'") {
      continue;
    } else {
      moves_array.push(moves[j]);
    }
  }
  disableButton("forwardButton");
  disableButton("rewindButton");
  setTimeout(() => {
    //enableButton("forwardButton");
    enableButton("rewindButton");
    enableSpeedSlider();
  }, getAnimationTime() * 2.3 * moves_array.length);
  currentMoves = moves_array;
  console.log(moves_array);

  // Delay each move by 1000 milliseconds
  moves_array.forEach((i, k) => {
    setTimeout(() => {
      itemRunner(i);
    }, k * getAnimationTime() * 2);
  });
}

function delay() {
  return new Promise(function (resolve) {
    setTimeout(resolve, getAnimationTime() / 2);
  });
}

/* you actual processing function */
async function itemRunner(i) {
  await delay();
  //console.log(i);

  // Front face moves
  if (i == "f") {
    SetFrontC(true);
    setTimeout(() => {
      SetFrontC(false);
    }, getAnimationTime());
    console.log("Front face - clockwise");
    turnZ(1, 0);
  } else if (i == "f'") {
    SetFrontCC(true);
    setTimeout(() => {
      SetFrontCC(false);
    }, getAnimationTime());
    console.log("Front face - counter");
    turnZ(1, 1);
  }
  // Back face move
  else if (i == "b'") {
    SetBackCC(true);
    setTimeout(() => {
      SetBackCC(false);
    }, getAnimationTime());
    console.log("Back face - counter");
    turnZ(-1, 0);
  } else if (i == "b") {
    SetBackC(true);
    setTimeout(() => {
      SetBackC(false);
    }, getAnimationTime());
    console.log("Back face - clockwise");
    turnZ(-1, 1);
  }
  // Right face move
  else if (i == "r") {
    SetRightC(true);
    setTimeout(() => {
      SetRightC(false);
    }, getAnimationTime());
    console.log("Right face - clockwise");
    turnX(1, 0);
  } else if (i == "r'") {
    SetRightCC(true);
    setTimeout(() => {
      SetRightCC(false);
    }, getAnimationTime());
    console.log("Right face - counter");
    turnX(1, 1);
  }
  // Left face move
  else if (i == "l'") {
    SetLeftCC(true);
    setTimeout(() => {
      SetLeftCC(false);
    }, getAnimationTime());
    console.log("Left face - counter");
    turnX(-1, 0);
  } else if (i == "l") {
    SetLeftC(true);
    setTimeout(() => {
      SetLeftC(false);
    }, getAnimationTime());
    console.log("Left face - clockwise");
    turnX(-1, 1);
  }
  // Up face move
  else if (i == "u'") {
    SetUpCC(true);
    setTimeout(() => {
      SetUpCC(false);
    }, getAnimationTime());
    console.log("Up face - counter");
    turnY(1, 0);
  } else if (i == "u") {
    SetUpC(true);
    setTimeout(() => {
      SetUpC(false);
    }, getAnimationTime());
    console.log("Up face - clockwise");
    turnY(1, 1);
  }
  // Down face move
  else if (i == "d") {
    SetDownC(true);
    setTimeout(() => {
      SetDownC(false);
    }, getAnimationTime());
    console.log("Down face - clockwise");
    turnY(-1, 0);
  } else if (i == "d'") {
    SetDownCC(true);
    setTimeout(() => {
      SetDownCC(false);
    }, getAnimationTime());
    console.log("Down face - counter");
    turnY(-1, 1);
  }

    updateState(i, false);
}

export function getRandomMove() {
  var moves = [-1, 1];
  var num = Math.floor(Math.random() * moves.length);
  console.log("Random result: ", num);
  return moves[num];
}
