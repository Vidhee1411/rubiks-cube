import { addArrows } from "../arrows";
import {
  activateMirrors,
  assignColors,
  deactivateMirrors,
  generateAllMirrors,
  highlightCornerPiece,
  highlightEdgePiece,
  highlightFrontFace,
  highlightMirror1,
  highlightMirror2,
  highlightMirror3,
  highlightRightFace,
  highlightSticker,
  highlightTopFace,
  makeHighlight,
  removeCornerPieceHighlight,
  removeEdgePieceHighlight,
  removeFrontFaceHighlight,
  removeFullCubeHighlight,
  removeMirror1Highlight,
  removeMirror2Highlight,
  removeMirror3Highlight,
  removeRightFaceHighlight,
  removeStickerHighlight,
  removeTopFaceHighlight,
  resetCubeConfiguration,
} from "../cube";
import { setOnClickFunctions } from "../Player";
import { enableChatbot, setUIButtons } from "./utilsUI";
import { generateCube } from "../cube";
import Swal from "sweetalert2";
import { performMoves } from "../keyListeners";
import { applyState } from "../state";
import { setMoveButtons } from "../controls";
export var pageInitialized = false;

export function guide(firstTime = true) {
  /**
   * For Basic Run
   */
  //   generateCube();
  //   assignColors();
  //   addArrows();
  //   generateAllMirrors();
  //   setOnClickFunctions();
  //   setUIButtons();
  /************************************************ */
  /**
   * For Basic Run
   */

  // if (firstTime) {
  //   generateCube();
  //   assignColors();
  //   addArrows();
  //   generateAllMirrors();
  // }
  // login();
  Swal.fire({
    title: "Welcome to ALLURE",
    position: "top",
    confirmButtonText: "Cool",
    allowOutsideClick: false,
  }).then((result) => {
    Swal.fire({
      title: "Do you want go through the tutorial?",
      showDenyButton: true,
      confirmButtonText: "Yep",
      denyButtonText: "Nope",
      icon: "question",
      position: "top",
    }).then((result) => {
      /* Read more about isConfirmed, isDenied below */
      makeHighlight(3, 3, 3, 0, 0, 0, 1.2);
      if (result.isConfirmed) {
        Swal.fire({
          title: "Introduction",
          text: "Awesome. So this is the Rubik's cube you will be using",
          position: "top",
          confirmButtonText: "Sounds good!",
          allowOutsideClick: false,
        }).then((result) => {
          /* Read more about isConfirmed, isDenied below */
          removeFullCubeHighlight();
          highlightMirror1();
          highlightMirror2();
          highlightMirror3();

          Swal.fire({
            title: "Rubik's Cube Faces",
            text: "The highlighted sides are the mirrors showing LEFT, BACK, and DOWN sides",
            position: "top",
            confirmButtonText: "Sounds good!",
            allowOutsideClick: false,
          }).then((result) => {
            resetCubeConfiguration();
            teachFaces();
          });
        });
      } else if (result.isDenied) {
        enableChatbot();
        resetCubeConfiguration();
        removeFullCubeHighlight();
      }
    });
  });
  doDemoMoves();
  setOnClickFunctions();
  setMoveButtons();
  setUIButtons();
  //disableAllElements();
}

function disableAllElements() {
  try {
    deactivateMirrors();
  } catch (e) {}
  disablePlayerButtons();
  disableInfoButtons();
}
function enableAllElements() {
  enablePlayerButtons();
  activateMirrors();
  enableInfoButtons();
}

function enableElement(id) {
  document.getElementById(id).style.display = "block";
}
function disableElement(id) {
  document.getElementById(id).style.display = "none";
}

function disablePlayerButtons() {
  disableElement("rewindButton");
  disableElement("forwardButton");
  disableElement("resetButton");
}

function disableInfoButtons() {
  disableElement("movesButton");
  disableElement("infoButton");
  disableElement("infoMAButton");
}

function enablePlayerButtons() {
  enableElement("rewindButton");
  enableElement("forwardButton");
  enableElement("resetButton");
}

function enableInfoButtons() {
  enableElement("movesButton");
  enableElement("infoButton");
  enableElement("infoMAButton");
}

function doDemoMoves() {
  resetCubeConfiguration();
  performMoves([], "ff'");
  pageInitialized = true;
}

export function teachFaces() {
  /* Read more about isConfirmed, isDenied below */
  removeMirror1Highlight();
  removeMirror2Highlight();
  removeMirror3Highlight();
  highlightMirror1();
  Swal.fire({
    title: "Faces",
    text: "Highlighted side is the LEFT face",
    position: "bottom-start",
    confirmButtonText: "Ok",
    allowOutsideClick: false,
  }).then((result) => {
    /* Read more about isConfirmed, isDenied below */
    removeMirror1Highlight();
    highlightMirror2();
    Swal.fire({
      title: "Faces",
      text: "That's the BACK face",
      position: "bottom-start",
      confirmButtonText: "Alright",
      allowOutsideClick: false,
    }).then((result) => {
      /* Read more about isConfirmed, isDenied below */
      removeMirror2Highlight();
      highlightMirror3();
      Swal.fire({
        title: "Faces",
        text: "This is the DOWN face",
        position: "bottom-start",
        confirmButtonText: "Got it",
        allowOutsideClick: false,
      }).then((result) => {
        /* Read more about isConfirmed, isDenied below */
        removeMirror3Highlight();
        highlightFrontFace();
        Swal.fire({
          title: "Faces",
          text: "That's the FRONT face",
          position: "bottom-start",
          confirmButtonText: "Alright",
          allowOutsideClick: false,
        }).then((result) => {
          /* Read more about isConfirmed, isDenied below */
          removeFrontFaceHighlight();
          highlightRightFace();
          Swal.fire({
            title: "Faces",
            text: "This is the RIGHT face",
            position: "bottom-start",
            confirmButtonText: "Ok",
            allowOutsideClick: false,
          }).then((result) => {
            /* Read more about isConfirmed, isDenied below */
            removeRightFaceHighlight();
            highlightTopFace();
            Swal.fire({
              title: "Faces",
              text: "Finally, that's the TOP face",
              position: "bottom-start",
              confirmButtonText: "Ok",
              allowOutsideClick: false,
            }).then((result) => {
              /* Read more about isConfirmed, isDenied below */
              removeTopFaceHighlight();
              highlightCornerPiece();
              Swal.fire({
                title: "Pieces",
                text: "This is a corner piece",
                position: "bottom-start",
                confirmButtonText: "Ok",
                allowOutsideClick: false,
              }).then((result) => {
                /* Read more about isConfirmed, isDenied below */
                removeCornerPieceHighlight();
                highlightEdgePiece();
                Swal.fire({
                  title: "Pieces",
                  text: "This is an edge piece",
                  position: "bottom-start",
                  confirmButtonText: "Ok",
                  allowOutsideClick: false,
                }).then((result) => {
                  /* Read more about isConfirmed, isDenied below */
                  removeEdgePieceHighlight();
                  highlightSticker();
                  Swal.fire({
                    title: "Pieces",
                    text: "And that's a sticker",
                    position: "bottom-start",
                    confirmButtonText: "Ok",
                    allowOutsideClick: false,
                  }).then((result) => {
                    /* Read more about isConfirmed, isDenied below */
                    removeStickerHighlight();
                    resetCubeConfiguration();
                    teachMoves();
                  });
                });
              });
            });
          });
        });
      });
    });
  });
}

export function teachMoves() {
  Swal.fire({
    title: "Moves",
    text: "Let's learn some terminologies of basic moves you can perform",
    position: "bottom-start",
    confirmButtonText: "Ok",
    allowOutsideClick: false,
  }).then((result) => {
    Swal.fire({
      title: "Moves",
      text: "This will be the Front Face Clockwise move aka F. Click play to perform this move.",
      position: "bottom-start",
      confirmButtonText: "Play",
      allowOutsideClick: false,
    }).then((result) => {
      /* Read more about isConfirmed, isDenied below */
      performMoves([], "f");
      Swal.fire({
        title: "Moves",
        text: "That was the Front Face Clockwise move aka F.",
        position: "bottom-start",
        confirmButtonText: "Ok",
        allowOutsideClick: false,
      }).then((result) => {
        /* Read more about isConfirmed, isDenied below */
        Swal.fire({
          title: "Moves",
          text: "This next will be the Back Face Counter-Clockwise move aka B'. Click play to perform this move.",
          position: "bottom-start",
          confirmButtonText: "Play",
          allowOutsideClick: false,
        }).then((result) => {
          /* Read more about isConfirmed, isDenied below */
          performMoves([], "b'");
          Swal.fire({
            title: "Moves",
            text: "That was the Back Face Counter-Clockwise move aka B'.",
            position: "bottom-start",
            confirmButtonText: "Ok",
            allowOutsideClick: false,
          }).then((result) => {
            /* Read more about isConfirmed, isDenied below */
            resetCubeConfiguration();
            chatBotInstructions();
          });
        });
      });
    });
  });
}

export function chatBotInstructions() {
  Swal.fire({
    title: "Chatbot",
    text: "Open up the chat on the bottom right and let's learn together. Let's begin",
    // confirmButtonText: "Tell me more",
    confirmButtonText: "Okay! Show me the chatbot",
    position: "bottom-end",
  }).then((result) => {
    /* Read more about isConfirmed, isDenied below */
    enableChatbot();
    resetCubeConfiguration();
    // if (result.isConfirmed) {
    //   Swal.fire({
    //     title: "Chatbot",
    //     text: 'In order to move the cube there is a set of 9 levels that . These commands are called "macro actions" Each macro action will be followed by a number(1-9) which has a specific move tied to the algorithm.',
    //     confirmButtonText: "Sounds good!",
    //     position: "bottom-end",
    //     allowOutsideClick: "false",
    //   }).then((result) => {
    //     resetCubeConfiguration();
    //   });
    // }
  });
}

export function login() {
  Swal.fire({
    title: "Login Form",
    html: `<input type="text" id="login" class="swal2-input" placeholder="Username">
    <input type="password" id="password" class="swal2-input" placeholder="Password">`,
    confirmButtonText: "Sign in",
    focusConfirm: false,
    preConfirm: () => {
      const login = Swal.getPopup().querySelector("#login").value;
      const password = Swal.getPopup().querySelector("#password").value;
      if (!login || !password) {
        Swal.showValidationMessage(`Please enter login and password`);
      }
      return { login: login, password: password };
    },
  }).then((result) => {
    Swal.fire(
      `
      Login: ${result.value.login}
      Password: ${result.value.password}
    `.trim()
    );
  });
}
