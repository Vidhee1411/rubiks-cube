import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import {
  resetToOriginalColors,
  assignColors,
  generateCube,
  generateAllMirrors,
  grayCubieAtXYZOnFaces,
  resetScene,
  turnY,
  deactivateMirrors,
  activateMirrors,
  getAnimationTime,
  disableSpeedSlider,
  cubeArray
} from "./cube";
import { onDocumentKeyDown, performMoves, getCurrentMoves } from "./keyListeners";
import { TWEEN } from "./utils";
import { app } from "./firebaseCred";
import { getDatabase, ref, onValue, set, update } from "firebase/database";
import { applyState, updateState } from "./state";
import { currentMoves, disableButton, enableButton, setCurrentMoves, setCurrentState, setOnClickFunctions } from "./Player";
import { addArrows } from "./arrows";
import { FirebaseError } from "@firebase/util";
import { setUIButtons } from "./UIGuide/utilsUI";
import { guide, pageInitialized } from "./UIGuide/guide";
import { getFirestore, collection, addDoc } from "firebase/firestore";
import { 
  getAuth,
  onAuthStateChanged, 
  signOut,
  signInWithEmailAndPassword,
} from 'firebase/auth';
// import { Firebase } from 'firebase'
export var scene, renderer, camera, controls;

export var rightFace = [0, 1];
export var leftFace = [2, 3];
export var topFace = [4, 5];
export var bottomFace = [6, 7];
export var frontFace = [8, 9];
export var backFace = [10, 11];
var state = [];

export var pageNotInitialized = false;

export function getCurrentState() {
  return state;
}

export function updateSolveState() {
  update(ref(getDatabase(app), '/user1'),{
        solve: "true"
      }).then(() => {
        // Data saved successfully!
        console.log("data saved");
      })
          .catch((error) => {
            // The write failed...
            console.log("data save failed");
          });
}

const database = getDatabase(app);
const auth = getAuth(app);
var user = "user1";
const stateListener = ref(database, user + "/state");
onValue(stateListener, (snapshot) => {
  const data = snapshot.val();
  state = data;
  if (data != "null") {
    resetCameraPosition();
    applyState(data);
    updateState('', true);
    pageNotInitialized = true;
  }
  console.log(data);
});

const solveListener = ref(database, user + "/solve");
onValue(solveListener, (snapshot) => {
  const data = snapshot.val();
  console.log("Solve data: ", data);
    if (data == "true") {
        pageNotInitialized = false;
    var moves;
    var moves_array;
    const movesListener = ref(database, user + "/moves");
    onValue(movesListener, (snapshot) => {
      const data2 = snapshot.val();
      if (data2 != "null") {
        moves = data2;
        moves_array = [];
        disableSpeedSlider();
      }
    });
        if (moves != "null") {
            setTimeout(() => {
                console.log("Moves are being performed");
                console.log("Moves that will occur: ", moves);
                performMoves(moves_array, moves);
                setCurrentMoves(getCurrentMoves());
                setCurrentState(getCurrentState());
            }, getAnimationTime() / 2);
        }
  }
});

init();

export function init() {

  
  renderer = new THREE.WebGLRenderer({ antialias: true });

  /*************************************BOILERPLATE CODE STARTS************************************************** */
  renderer.setPixelRatio(window.devicePixelRatio); //this is to get the correct pixel detail on portable devices
  renderer.setSize(window.innerWidth, window.innerHeight); //and this sets the canvas' size.
  document.body.appendChild(renderer.domElement);
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x008200 + 200);
  const axesHelper = new THREE.AxesHelper(5);
  /*Enable the line below to create a axis to help in development*/
  //scene.add(axesHelper);
  camera = new THREE.PerspectiveCamera(
    75, //FOV
    window.innerWidth / window.innerHeight, //aspect
    1, //near clipping plane
    100 //far clipping plane
  );
  // Load the background texture
  //Load background texture
  var texture = new THREE.TextureLoader().load("https://cdn.pixabay.com/photo/2019/01/02/19/29/background-3909535_960_720.jpg");
  scene.background = texture;
  camera.position.set(6.5, 6.5, 6.5);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.rotateSpeed = 0.2;
  controls.enableDamping = false;
  controls.dampingFactor = 0.05;
  controls.enableZoom = false;
  controls.enabled = true;
  window.addEventListener("resize", function () {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });
  /*************************************BOILERPLATE CODE ENDS************************************************** */

  /*********************CREATION OF CUBE, INITIAL COLOR ASSIGNING, AND GENERATING MIRRORS*********************** */
  animate();
  // guide();
  generateCube();
  assignColors();
  addArrows();
  generateAllMirrors();
}

export function recreate() {
  generateCube();
  assignColors();
  generateAllMirrors();
  /*********************CREATION OF CUBE, INITIAL COLOR ASSIGNING, AND GENERATING MIRRORS*********************** */

  grayCubieAtXYZOnFaces(1, 1, 1, frontFace);
  grayCubieAtXYZOnFaces(1, 0, 1, frontFace);
  resetToOriginalColors();
  animate();
}

function animate() {
  TWEEN.update();
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

export function resetCameraPosition() {
  activateMirrors();
  camera.position.set(6.5, 6.5, 6.5);
}

controls.addEventListener("change", () => {
  if (camera.position.x != 6.5 && camera.position.y != 6.5 && camera.position.z != 6.5) deactivateMirrors();
});
//document.addEventListener("keydown", onDocumentKeyDown, false);

//var toggle = document.getElementById("showBtn")



var login = document.getElementById("login");
var txtEmail = document.getElementById("txtEmail");
var txtPassword = document.getElementById("txtPassword");
var startingTime = null;

const loginEmailPassword = async () => {

  console.log("Login attempting")
  const loginEmail = txtEmail.value
  const loginPassword = txtPassword.value

  try {
    await signInWithEmailAndPassword(auth, loginEmail, loginPassword)
    startingTime = new Date();
  }
  catch(error) {
    console.log(`There was an error: ${error}`)
  }




}

var endingTime = null;
function logoutUser() {
  signOut(auth).then(() => {
    endingTime = new Date();
    console.log("User Signed out")
    console.log("Current user: ", auth.currentUser)
    myForm.style.display = 'block';
    writeToFB();
    var i;

    console.log("local storage");
    for (i = 0; i < localStorage.length; i++)   {
      console.log(localStorage.key(i) + "=[" + localStorage.getItem(localStorage.key(i)) + "]");
    }
  }).catch((error) => {
      console.log(`There was an error: ${error}`)
  });
  
}
auth.onAuthStateChanged(function(user) {
  if (user) {
    // if (user. === false) {
    //   auth.signOut()
    // }
    // else{
    myForm.style.display = "none";
    console.log("Login successful");
    const uid = auth.currentUser;
    console.log("Current user: ", uid)
    guide();
    // }
  }
  else{
    console.log("Something else happened")
  }
});
const showButton = () => {
  console.log("Button pressed")
  toggle.style.display = "block";
}
const hideButton = () => {
  toggle.style.display = "none";
}

function openForm() {
  document.getElementById("myForm").style.display = "block";
}

function closeForm() {
  document.getElementById("myForm").style.display = "none";
}

// Count the number of times info is clicked
localStorage.setItem("Info Click Count", 0);
var infoArray = [];
function addInfoClick(){
  var currentClicks = localStorage.getItem("Info Click Count")
  var currentDate = new Date();
  currentClicks++;

  let click = {
    clickNumber: currentClicks,
    timestamp: currentDate
  };
  infoArray.push(click);
  localStorage.setItem("Info Click/Time", JSON.stringify(infoArray));
  localStorage.setItem("Info Click Count", currentClicks);
}

// Count the number of times levels is clicked
localStorage.setItem("Levels Click Count", 0);
var levelsArray = [];
function addLevelsClick(){
  console.log("Levels clicked")
  var currentClicks = localStorage.getItem("Levels Click Count")
  var currentDate = new Date();
  currentClicks++;

  let click = {
    clickNumber: currentClicks,
    timestamp: currentDate
  };
  levelsArray.push(click);
  localStorage.setItem("Levels Click/Time", JSON.stringify(levelsArray));
  localStorage.setItem("Levels Click Count", currentClicks);
}

// Count the number of times moves is clicked
localStorage.setItem("Moves Click Count", 0);
var movesArray = [];
function addMovesClick(){
  console.log("Moves clicked")
  var currentClicks = localStorage.getItem("Moves Click Count")
  var currentDate = new Date();
  currentClicks++;

  let click = {
    clickNumber: currentClicks,
    timestamp: currentDate
  };
  movesArray.push(click);
  localStorage.setItem("Moves Click/Time", JSON.stringify(movesArray));
  localStorage.setItem("Moves Click Count", currentClicks);
}

// Count the number of times user presses complete instructions button
localStorage.setItem("Complete Intructions Click Count", 0);
var completeInsArray = [];
function addCompleteInsClick(){
  console.log("Complete Intructions clicked")
  var currentClicks = localStorage.getItem("Complete Instructions Click Count")
  var currentDate = new Date();
  currentClicks++;

  let click = {
    clickNumber: currentClicks,
    timestamp: currentDate
  };
  completeInsArray.push(click);
  localStorage.setItem("Complete Instructions Click/Time", JSON.stringify(completeInsArray));
  localStorage.setItem("Complete Instructions Click Count", currentClicks);
}

// Count the number of times user presses faces instructions button
localStorage.setItem("Faces Intructions Click Count", 0);
var facesInsArray = [];
function addFacesInsClick(){
  console.log("Face Intructions clicked")
  var currentClicks = localStorage.getItem("Faces Instructions Click Count")
  var currentDate = new Date();
  currentClicks++;

  let click = {
    clickNumber: currentClicks,
    timestamp: currentDate
  };
  facesInsArray.push(click);
  localStorage.setItem("Faces Instructions Click/Time", JSON.stringify(facesInsArray));
  localStorage.setItem("Faces Instructions Click Count", currentClicks);
}

//Count the number of times user presses chatbot instructions
localStorage.setItem("Chatbot Intructions Click Count", 0);
var chatbotInsArray = [];
function addChatbotInsClick(){
  var currentClicks = localStorage.getItem("Chatbot Instructions Click Count")
  var currentDate = new Date();
  currentClicks++;

  let click = {
    clickNumber: currentClicks,
    timestamp: currentDate
  };
  chatbotInsArray.push(click);
  localStorage.setItem("Chatbot Instructions Click/Time", JSON.stringify(chatbotInsArray));
  localStorage.setItem("Chatbot Instructions Click Count", currentClicks);
}

// Count rewind clicks
localStorage.setItem("Rewind Click Count", 0);
var rewindArray = [];
function addRewindClick(){
  console.log("Rewind clicked");
  var currentClicks = localStorage.getItem("Rewind Click Count")
  var currentDate = new Date();
  currentClicks++;

  let click = {
    clickNumber: currentClicks,
    timestamp: currentDate
  };
  rewindArray.push(click);
  localStorage.setItem("Rewind Click/Time", JSON.stringify(rewindArray));
  localStorage.setItem("Rewind Click Count", currentClicks);
}

// Count Reset clicks
localStorage.setItem("Reset Click Count", 0);
var resetArray = [];
function addResetClick(){
  console.log("Reset clicked");
  var currentClicks = localStorage.getItem("Reset Click Count")
  var currentDate = new Date();
  currentClicks++;

  let click = {
    clickNumber: currentClicks,
    timestamp: currentDate
  };
  resetArray.push(click);
  localStorage.setItem("Reset Click/Time", JSON.stringify(resetArray));
  localStorage.setItem("Reset Click Count", currentClicks);
}

//Added log for reset cube button
// Count Reset Cube clicks
localStorage.setItem("Reset Cube Click Count", 0);
var resetCubeArray = [];
function addResetCubeClick(){
  console.log("Reset Cube clicked");
  var currentClicks = localStorage.getItem("Reset Cube Click Count")
  var currentDate = new Date();
  currentClicks++;

  let click = {
    clickNumber: currentClicks,
    timestamp: currentDate
  };
  resetCubeArray.push(click);
  localStorage.setItem("Reset Cube Click/Time", JSON.stringify(resetCubeArray));
  localStorage.setItem("Reset Cube Click Count", currentClicks);
}

// Count Forward clicks
localStorage.setItem("Forward Click Count", 0);
var forwardArray = [];
function addForwardClick(){
  console.log("Forward clicked");
  var currentClicks = localStorage.getItem("Forward Click Count")
  var currentDate = new Date();
  currentClicks++;

  let click = {
    clickNumber: currentClicks,
    timestamp: currentDate
  };
  forwardArray.push(click);
  localStorage.setItem("Forward Click/Time", JSON.stringify(forwardArray));
  localStorage.setItem("Forward Click Count", currentClicks);
}

// Slider changes
var sliderVal = document.getElementById("speedSlider");
var sliderArray = [];
function sliderChange(){
  console.log("Slider Moved");
  var val = sliderVal.value;
  var currentDate = new Date();

  let click = {
    sliderValue: val,
    timestamp: currentDate
  };
  sliderArray.push(click);
  localStorage.setItem("Slider Click/Time", JSON.stringify(sliderArray));
}

//Adding logs for the 12 moves buttons
// Count F clicks
localStorage.setItem("F Click Count", 0);
var moveFArray = [];
function addFcwClick() {
    console.log("F clicked");
    var currentClicks = localStorage.getItem("F Click Count")
    var currentDate = new Date();
    currentClicks++;

    let click = {
        clickNumber: currentClicks,
        timestamp: currentDate
    };
    moveFArray.push(click);
    localStorage.setItem("F Click/Time", JSON.stringify(moveFArray));
    localStorage.setItem("F Click Count", currentClicks);
}

// Count R clicks
localStorage.setItem("R Click Count", 0);
var moveRArray = [];
function addRcwClick() {
    console.log("R clicked");
    var currentClicks = localStorage.getItem("R Click Count")
    var currentDate = new Date();
    currentClicks++;

    let click = {
        clickNumber: currentClicks,
        timestamp: currentDate
    };
    moveRArray.push(click);
    localStorage.setItem("R Click/Time", JSON.stringify(moveRArray));
    localStorage.setItem("R Click Count", currentClicks);
}

// Count U clicks
localStorage.setItem("U Click Count", 0);
var moveUArray = [];
function addUcwClick() {
    console.log("U clicked");
    var currentClicks = localStorage.getItem("U Click Count")
    var currentDate = new Date();
    currentClicks++;

    let click = {
        clickNumber: currentClicks,
        timestamp: currentDate
    };
    moveUArray.push(click);
    localStorage.setItem("U Click/Time", JSON.stringify(moveUArray));
    localStorage.setItem("U Click Count", currentClicks);
}

// Count B clicks
localStorage.setItem("B Click Count", 0);
var moveBArray = [];
function addBcwClick() {
    console.log("B clicked");
    var currentClicks = localStorage.getItem("B Click Count")
    var currentDate = new Date();
    currentClicks++;

    let click = {
        clickNumber: currentClicks,
        timestamp: currentDate
    };
    moveBArray.push(click);
    localStorage.setItem("B Click/Time", JSON.stringify(moveBArray));
    localStorage.setItem("B Click Count", currentClicks);
}

// Count U clicks
localStorage.setItem("L Click Count", 0);
var moveLArray = [];
function addLcwClick() {
    console.log("U clicked");
    var currentClicks = localStorage.getItem("L Click Count")
    var currentDate = new Date();
    currentClicks++;

    let click = {
        clickNumber: currentClicks,
        timestamp: currentDate
    };
    moveLArray.push(click);
    localStorage.setItem("L Click/Time", JSON.stringify(moveLArray));
    localStorage.setItem("L Click Count", currentClicks);
}

// Count U clicks
localStorage.setItem("D Click Count", 0);
var moveDArray = [];
function addDcwClick() {
    console.log("D clicked");
    var currentClicks = localStorage.getItem("D Click Count")
    var currentDate = new Date();
    currentClicks++;

    let click = {
        clickNumber: currentClicks,
        timestamp: currentDate
    };
    moveDArray.push(click);
    localStorage.setItem("D Click/Time", JSON.stringify(moveDArray));
    localStorage.setItem("D Click Count", currentClicks);
}

// Count F' clicks
localStorage.setItem("F' Click Count", 0);
var moveFccwArray = [];
function addFccwClick() {
    console.log("F' clicked");
    var currentClicks = localStorage.getItem("F' Click Count")
    var currentDate = new Date();
    currentClicks++;
    let click = {
        clickNumber: currentClicks,
        timestamp: currentDate
    };
    moveFccwArray.push(click);
    localStorage.setItem("F' Click/Time", JSON.stringify(moveFccwArray));
    localStorage.setItem("F' Click Count", currentClicks);
}

// Count R' clicks
localStorage.setItem("R' Click Count", 0);
var moveRccwArray = [];
function addRccwClick() {
    console.log("R' clicked");
    var currentClicks = localStorage.getItem("R' Click Count")
    var currentDate = new Date();
    currentClicks++;

    let click = {
        clickNumber: currentClicks,
        timestamp: currentDate
    };
    moveRccwArray.push(click);
    localStorage.setItem("R' Click/Time", JSON.stringify(moveRccwArray));
    localStorage.setItem("R' Click Count", currentClicks);
}

// Count U' clicks
localStorage.setItem("U' Click Count", 0);
var moveUccwArray = [];
function addUccwClick() {
    console.log("U' clicked");
    var currentClicks = localStorage.getItem("U' Click Count")
    var currentDate = new Date();
    currentClicks++;

    let click = {
        clickNumber: currentClicks,
        timestamp: currentDate
    };
    moveUccwArray.push(click);
    localStorage.setItem("U' Click/Time", JSON.stringify(moveUccwArray));
    localStorage.setItem("U' Click Count", currentClicks);
}

// Count B' clicks
localStorage.setItem("B' Click Count", 0);
var moveBccwArray = [];
function addBccwClick() {
    console.log("B' clicked");
    var currentClicks = localStorage.getItem("B' Click Count")
    var currentDate = new Date();
    currentClicks++;

    let click = {
        clickNumber: currentClicks,
        timestamp: currentDate
    };
    moveBccwArray.push(click);
    localStorage.setItem("B' Click/Time", JSON.stringify(moveBccwArray));
    localStorage.setItem("B' Click Count", currentClicks);
}

// Count L' clicks
localStorage.setItem("L' Click Count", 0);
var moveLccwArray = [];
function addLccwClick() {
    console.log("L' clicked");
    var currentClicks = localStorage.getItem("L' Click Count")
    var currentDate = new Date();
    currentClicks++;

    let click = {
        clickNumber: currentClicks,
        timestamp: currentDate
    };
    moveLccwArray.push(click);
    localStorage.setItem("L' Click/Time", JSON.stringify(moveLccwArray));
    localStorage.setItem("L' Click Count", currentClicks);
}

// Count D' clicks
localStorage.setItem("D' Click Count", 0);
var moveDccwArray = [];
function addDccwClick() {
    console.log("D' clicked");
    var currentClicks = localStorage.getItem("D' Click Count")
    var currentDate = new Date();
    currentClicks++;

    let click = {
        clickNumber: currentClicks,
        timestamp: currentDate
    };
    moveDccwArray.push(click);
    localStorage.setItem("D' Click/Time", JSON.stringify(moveDccwArray));
    localStorage.setItem("D' Click Count", currentClicks);
}

//F hover
localStorage.setItem("F Mouseover Count", 0);
var moveTabFArray = [];
function addFcwMouseOver() {
    console.log("Mouse over F");
    var currentMouseOvers = localStorage.getItem("F Mouseover Count")
    var currentDate = new Date();
    currentMouseOvers++;

    let mouseOver = {
        mouseOverNumber: currentMouseOvers,
        timestamp: currentDate
    };
    moveTabFArray.push(mouseOver);
    localStorage.setItem("F Hover/Time", JSON.stringify(moveTabFArray));
    localStorage.setItem("F Mouseover Count", currentMouseOvers);
}

//function to capture event number and timestamp when mouse is moved off a particular element - in this case the F move in the moves tab
//Using the same array (moveTabFArray) to capture both mouseover and mouseout events for continuity
localStorage.setItem("F Mouseout Count", 0);
function addFcwMouseOut() {
    console.log("Mouse out F");
    var currentMouseOuts = localStorage.getItem("F Mouseout Count")
    var currentDate = new Date();
    currentMouseOuts++;

    let mouseOut = {
        mouseOutNumber: currentMouseOuts,
        timestamp: currentDate
    };
    moveTabFArray.push(mouseOut);

    //Using same item (F Hover/Time) for both mouseover and mouseout for continuity
    localStorage.setItem("F Hover/Time", JSON.stringify(moveTabFArray));
    localStorage.setItem("F Mouseout Count", currentMouseOuts);
}
/* //F' Hover
localStorage.setItem("F' Mouseover Count", 0);
var moveTabFccArray = [];
function addFccwMouseOver() {
    console.log("Mouse over F'");
    var currentMouseOvers = localStorage.getItem("F' Mouseover Count")
    var currentDate = new Date();
    currentMouseOvers++;

    let mouseOver = {
        mouseOverNumber: currentMouseOvers,
        timestamp: currentDate
    };
    moveTabFccArray.push(mouseOver);
    localStorage.setItem("F' Hover/Time", JSON.stringify(moveTabFccArray));
    localStorage.setItem("F' Mouseover Count", currentMouseOvers);
}
//B Hover
localStorage.setItem("B Mouseover Count", 0);
var moveTabBArray = [];
function addBcwMouseOver() {
    console.log("Mouse over B");
    var currentMouseOvers = localStorage.getItem("B Mouseover Count")
    var currentDate = new Date();
    currentMouseOvers++;

    let mouseOver = {
        mouseOverNumber: currentMouseOvers,
        timestamp: currentDate
    };
    moveTabBArray.push(mouseOver);
    localStorage.setItem("B Hover/Time", JSON.stringify(moveTabBArray));
    localStorage.setItem("B Mouseover Count", currentMouseOvers);
}

//B' Hover
localStorage.setItem("B' Mouseover Count", 0);
var moveTabBccArray = [];
function addBccwMouseOver() {
    console.log("Mouse over B'");
    var currentMouseOvers = localStorage.getItem("B' Mouseover Count")
    var currentDate = new Date();
    currentMouseOvers++;

    let mouseOver = {
        mouseOverNumber: currentMouseOvers,
        timestamp: currentDate
    };
    moveTabBccArray.push(mouseOver);
    localStorage.setItem("B' Hover/Time", JSON.stringify(moveTabBccArray));
    localStorage.setItem("B' Mouseover Count", currentMouseOvers);
}

//R hover
localStorage.setItem("R Mouseover Count", 0);
var moveTabRArray = [];
function addRcwMouseOver() {
    console.log("Mouse over R");
    var currentMouseOvers = localStorage.getItem("R Mouseover Count")
    var currentDate = new Date();
    currentMouseOvers++;

    let mouseOver = {
        mouseOverNumber: currentMouseOvers,
        timestamp: currentDate
    };
    moveTabRArray.push(mouseOver);
    localStorage.setItem("R Hover/Time", JSON.stringify(moveTabRArray));
    localStorage.setItem("R Mouseover Count", currentMouseOvers);
}
//R' Hover
localStorage.setItem("R' Mouseover Count", 0);
var moveTabRccArray = [];
function addRccwMouseOver() {
    console.log("Mouse over R'");
    var currentMouseOvers = localStorage.getItem("R' Mouseover Count")
    var currentDate = new Date();
    currentMouseOvers++;

    let mouseOver = {
        mouseOverNumber: currentMouseOvers,
        timestamp: currentDate
    };
    moveTabRccArray.push(mouseOver);
    localStorage.setItem("R' Hover/Time", JSON.stringify(moveTabRccArray));
    localStorage.setItem("R' Mouseover Count", currentMouseOvers);
}

//L hover
localStorage.setItem("L Mouseover Count", 0);
var moveTabLArray = [];
function addLcwMouseOver() {
    console.log("Mouse over L");
    var currentMouseOvers = localStorage.getItem("L Mouseover Count")
    var currentDate = new Date();
    currentMouseOvers++;

    let mouseOver = {
        mouseOverNumber: currentMouseOvers,
        timestamp: currentDate
    };
    moveTabLArray.push(mouseOver);
    localStorage.setItem("L Hover/Time", JSON.stringify(moveTabLArray));
    localStorage.setItem("L Mouseover Count", currentMouseOvers);
}
//L' Hover
localStorage.setItem("L' Mouseover Count", 0);
var moveTabLccArray = [];
function addLccwMouseOver() {
    console.log("Mouse over L'");
    var currentMouseOvers = localStorage.getItem("L' Mouseover Count")
    var currentDate = new Date();
    currentMouseOvers++;

    let mouseOver = {
        mouseOverNumber: currentMouseOvers,
        timestamp: currentDate
    };
    moveTabLccArray.push(mouseOver);
    localStorage.setItem("L' Hover/Time", JSON.stringify(moveTabLccArray));
    localStorage.setItem("L' Mouseover Count", currentMouseOvers);
}

//U hover
localStorage.setItem("U Mouseover Count", 0);
var moveTabUArray = [];
function addUcwMouseOver() {
    console.log("Mouse over U");
    var currentMouseOvers = localStorage.getItem("U Mouseover Count")
    var currentDate = new Date();
    currentMouseOvers++;

    let mouseOver = {
        mouseOverNumber: currentMouseOvers,
        timestamp: currentDate
    };
    moveTabUArray.push(mouseOver);
    localStorage.setItem("U Hover/Time", JSON.stringify(moveTabUArray));
    localStorage.setItem("U Mouseover Count", currentMouseOvers);
}
//U' Hover
localStorage.setItem("U' Mouseover Count", 0);
var moveTabUccArray = [];
function addUccwMouseOver() {
    console.log("Mouse over U'");
    var currentMouseOvers = localStorage.getItem("U' Mouseover Count")
    var currentDate = new Date();
    currentMouseOvers++;

    let mouseOver = {
        mouseOverNumber: currentMouseOvers,
        timestamp: currentDate
    };
    moveTabUccArray.push(mouseOver);
    localStorage.setItem("U' Hover/Time", JSON.stringify(moveTabUccArray));
    localStorage.setItem("U' Mouseover Count", currentMouseOvers);
}

//D hover
localStorage.setItem("D Mouseover Count", 0);
var moveTabDArray = [];
function addDcwMouseOver() {
    console.log("Mouse over D");
    var currentMouseOvers = localStorage.getItem("D Mouseover Count")
    var currentDate = new Date();
    currentMouseOvers++;

    let mouseOver = {
        mouseOverNumber: currentMouseOvers,
        timestamp: currentDate
    };
    moveTabDArray.push(mouseOver);
    localStorage.setItem("D Hover/Time", JSON.stringify(moveTabDArray));
    localStorage.setItem("D Mouseover Count", currentMouseOvers);
}
//D' Hover
localStorage.setItem("D' Mouseover Count", 0);
var moveTabDccArray = [];
function addDccwMouseOver() {
    console.log("Mouse over D'");
    var currentMouseOvers = localStorage.getItem("D' Mouseover Count")
    var currentDate = new Date();
    currentMouseOvers++;

    let mouseOver = {
        mouseOverNumber: currentMouseOvers,
        timestamp: currentDate
    };
    moveTabDccArray.push(mouseOver);
    localStorage.setItem("D' Hover/Time",JSON.stringify(moveTabDccArray));
    localStorage.setItem("D' Mouseover Count", currentMouseOvers);
}

 */
const db = getFirestore(app);

const writeToFB = async () => {

  try {
    const docRef = await addDoc(collection(db, "User_testing"), {
      user: txtEmail.value,
      startTime: startingTime,
      endTime: endingTime,
      infoClicks: JSON.parse(localStorage.getItem("Info Click/Time")),
      moveClicks: JSON.parse(localStorage.getItem("Moves Click/Time")),
      levelsClicks: JSON.parse(localStorage.getItem("Levels Click/Time")),
      completeInsClicks: JSON.parse(localStorage.getItem("Complete Instructions Click/Time")),
      levelsInsClicks: JSON.parse(localStorage.getItem("Levels Instructions Click/Time")),
      chatbotInsClicks: JSON.parse(localStorage.getItem("Chatbot Instructions Click/Time")),
      conversation: JSON.parse(localStorage.getItem("chat_session")),
      rewindClicks: JSON.parse(localStorage.getItem("Rewind Click/Time")),
      resetClicks: JSON.parse(localStorage.getItem("Reset Click/Time")),
      resetCubeClicks: JSON.parse(localStorage.getItem("Reset Cube Click/Time")),
      forwardClicks: JSON.parse(localStorage.getItem("Forward Click/Time")),
      sliderChanges: JSON.parse(localStorage.getItem("Slider Click/Time")),

      //Adding move button clicks to firebase
      fcwClicks: JSON.parse(localStorage.getItem("F Click/Time")),
      rcwClicks: JSON.parse(localStorage.getItem("R Click/Time")),
      ucwClicks: JSON.parse(localStorage.getItem("U Click/Time")),
      bcwClicks: JSON.parse(localStorage.getItem("B Click/Time")),
      lcwClicks: JSON.parse(localStorage.getItem("L Click/Time")),
      dcwClicks: JSON.parse(localStorage.getItem("D Click/Time")),
      fccwClicks: JSON.parse(localStorage.getItem("F' Click/Time")),
      rccwClicks: JSON.parse(localStorage.getItem("R' Click/Time")),
      uccwClicks: JSON.parse(localStorage.getItem("U' Click/Time")),
      bccwClicks: JSON.parse(localStorage.getItem("B' Click/Time")),
      lccwClicks: JSON.parse(localStorage.getItem("L' Click/Time")),
      dccwClicks: JSON.parse(localStorage.getItem("D' Click/Time")),
      
      //adding hovering to firebase
      fcwHovers: JSON.parse(localStorage.getItem("F Hover/Time")),
      /* fccwHovers: JSON.parse(localStorage.getItem("F' Hover/Time")),
      bcwHovers: JSON.parse(localStorage.getItem("B Hover/Time")),
      bccwHovers: JSON.parse(localStorage.getItem("B' Hover/Time")),
      rcwHovers: JSON.parse(localStorage.getItem("R Hover/Time")),
      rccwHovers: JSON.parse(localStorage.getItem("R' Hover/Time")),
      lcwHovers: JSON.parse(localStorage.getItem("L Hover/Time")),
      lccwHovers: JSON.parse(localStorage.getItem("L' Hover/Time")),
      ucwHovers: JSON.parse(localStorage.getItem("U Hover/Time")),
      uccwHovers: JSON.parse(localStorage.getItem("U' Hover/Time")),
      dcwHovers: JSON.parse(localStorage.getItem("D Hover/Time")),
      dccwHovers: JSON.parse(localStorage.getItem("D' Hover/Time")),
     */});
    console.log("Document written with ID: ", docRef.id);
  } catch (e) {
    console.error("Error adding document: ", e);
  }
  

}

// Listeners for rewind, reset, reset cube, and forward
document.getElementById("rewindButton").addEventListener("click", addRewindClick);
document.getElementById("resetButton").addEventListener("click", addResetClick);
document.getElementById("resetCubeButton").addEventListener("click", addResetCubeClick);
document.getElementById("forwardButton").addEventListener("click", addForwardClick);

// Listeners for info, levels, and moves buttons
document.getElementById("infoBtn").addEventListener("click", addInfoClick);
document.getElementById("levelsBtn").addEventListener("click", addLevelsClick);
document.getElementById("movesBtn").addEventListener("click", addMovesClick);

// Listeners for tutorial buttons
document.getElementById("completeInstructionButton").addEventListener("click", addCompleteInsClick);
document.getElementById("facesInstructionButton").addEventListener("click", addFacesInsClick);
document.getElementById("instructionButton").addEventListener("click", addChatbotInsClick);

// Slider listener
document.getElementById("speedSlider").addEventListener("change", sliderChange)

// Login and Logout Listeners
document.getElementById("btnLogin").addEventListener("click", loginEmailPassword);
document.getElementById("btnLogout").addEventListener("click", logoutUser);

//Listeners for the 12 move buttons - F,R,U,B,L,D and F',R',U',B',L',D'
document.getElementById("f_cw").addEventListener("click", addFcwClick);
document.getElementById("r_cw").addEventListener("click", addRcwClick);
document.getElementById("u_cw").addEventListener("click", addUcwClick);
document.getElementById("b_cw").addEventListener("click", addBcwClick);
document.getElementById("l_cw").addEventListener("click", addLcwClick);
document.getElementById("d_cw").addEventListener("click", addDcwClick);

document.getElementById("f_ccw").addEventListener("click", addFccwClick);
document.getElementById("r_ccw").addEventListener("click", addRccwClick);
document.getElementById("u_ccw").addEventListener("click", addUccwClick);
document.getElementById("b_ccw").addEventListener("click", addBccwClick);
document.getElementById("l_ccw").addEventListener("click", addLccwClick);
document.getElementById("d_ccw").addEventListener("click", addDccwClick);

//moves tab on left
document.getElementById("mb_f_cw").addEventListener("mouseover", addFcwMouseOver);
document.getElementById("mb_f_cw").addEventListener("mouseout", addFcwMouseOut);
/* document.getElementById("mb_f_ccw").addEventListener("mouseover", addFccwMouseOver);
document.getElementById("mb_b_cw").addEventListener("mouseover", addBcwMouseOver);
document.getElementById("mb_b_ccw").addEventListener("mouseover", addBccwMouseOver);
document.getElementById("mb_r_cw").addEventListener("mouseover", addRcwMouseOver);
document.getElementById("mb_r_ccw").addEventListener("mouseover", addRccwMouseOver);
document.getElementById("mb_l_cw").addEventListener("mouseover", addLcwMouseOver);
document.getElementById("mb_l_ccw").addEventListener("mouseover", addLccwMouseOver);
document.getElementById("mb_u_cw").addEventListener("mouseover", addUcwMouseOver);
document.getElementById("mb_u_ccw").addEventListener("mouseover", addUccwMouseOver);
document.getElementById("mb_d_cw").addEventListener("mouseover", addDcwMouseOver);
document.getElementById("mb_d_ccw").addEventListener("mouseover", addDccwMouseOver);
 */ 