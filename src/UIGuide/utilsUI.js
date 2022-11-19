import infoButtonImage from "./../images/infoButton.png";
import bulbImage from "./../images/bulb.png";
import movesImage from "./../images/moves.png";
import level11 from "./../images/Levels/level1/1.1/level11.png";
import level11firstF from "./../images/Levels/level1/1.1/level11firstF.png";
import level11secondF from "./../images/Levels/level1/1.1/level11secondF.png";
import level12 from "./../images/Levels/level1/1.2/level12.png";
import level12Fdash from "./../images/Levels/level1/1.2/level12Fdash.png";
import level21 from "./../images/Levels/level2/2.1/level21.png";
import level21D from "./../images/Levels/level2/2.1/level21D.png";
import level22 from "./../images/Levels/level2/2.2/level22.png";
import level22Ddash from "./../images/Levels/level2/2.2/level22Ddash.png";
import level22DdashSecond from "./../images/Levels/level2/2.2/level22DdashSecond.png";
import level31 from "./../images/Levels/level3/3.1/level31.png";
import level31L from "./../images/Levels/level3/3.1/level31L.png";
import level32 from "./../images/Levels/level3/3.2/level32.png";
import level32Rdash from "./../images/Levels/level3/3.2/level32Rdash.png";
import level41 from "./../images/Levels/Level4/4.1/level41.png";
import level41R from "./../images/Levels/Level4/4.1/level41R.png";
import level41Rdash from "./../images/Levels/Level4/4.1/level41Rdash.png";
import level41U from "./../images/Levels/Level4/4.1/level41U.png";
import level41Udash from "./../images/Levels/Level4/4.1/level41Udash.png";
import level42 from "./../images/Levels/level4/4.2/level42.png";
import level42L from "./../images/Levels/Level4/4.2/level42L.png";
import level42U from "./../images/Levels/Level4/4.2/level42U.png";
import level42Ldash from "./../images/Levels/Level4/4.2/level42Ldash.png";
import level42Udash from "./../images/Levels/Level4/4.2/level42Udash.png";
import level5 from "./../images/Levels/Level5/level5.png";
import level5D from "./../images/Levels/Level5/level5D.png";
import level5Fdash from "./../images/Levels/Level5/level5Fdash.png";
import level5R from "./../images/Levels/Level5/level5R.png";
import level5Rdash from "./../images/Levels/Level5/level5Rdash.png";
import level6 from "./../images/Levels/Level6/level6.png";
import level6F from "./../images/Levels/Level6/level6F.png";
import level6R from "./../images/Levels/Level6/level6R.png";
import level6U from "./../images/Levels/Level6/level6U.png";
import level6Udash from "./../images/Levels/Level6/level6Udash.png";
import level7 from "./../images/Levels/Level7/level7.png";
import level7B from "./../images/Levels/Level7/level7B.png";
import level7L from "./../images/Levels/Level7/level7L.png";
import level7U from "./../images/Levels/Level7/level7U.png";
import level7Udash from "./../images/Levels/Level7/level7Udash.png";
import level81 from "./../images/Levels/Level8/8.1/level81.png";
import level81DFirstdash from "./../images/Levels/Level8/8.1/level81DFirstdash.png";
import level81DSeconddash from "./../images/Levels/Level8/8.1/level81DSeconddash.png";
import level9 from "./../images/Levels/Level9/level9.png";
import level9B from "./../images/Levels/Level9/level9B.png";
import level9L from "./../images/Levels/Level9/level9L.png";
import level9secondL from "./../images/Levels/Level9/level9Lsecond.png";
import level9U from "./../images/Levels/Level9/level9U.png";
import level9Udash from "./../images/Levels/Level9/level9Udash.png";



import Swal from "sweetalert2";
import { chatBotInstructions, guide, teachFaces } from "./guide";
export function setUIButtons() {
  setUIButtonsImages();
  document.getElementById("infoButton").onclick = function () {
    infoButtonShowHide("infoContainer");
  };
  document.getElementById("infoMAButton").onclick = function () {
    infoButtonShowHide("mAInfoContainer");
  };
  document.getElementById("movesButton").onclick = function () {
    infoButtonShowHide("movesContainer");
  };
  document.getElementById("instructionButton").onclick = function () {
    chatBotInstructions();
  };
  document.getElementById("facesInstructionButton").onclick = function () {
    teachFaces();
  };
  document.getElementById("completeInstructionButton").onclick = function () {
    guide(false);
  };

  document.getElementById("confButton").onclick = function () {
    console.log("Conf clicked");
    var e = document.getElementById("color-grid");
    if (e.style.display == "none") e.style.display = "grid";
    else e.style.display = "none";
  };
}
function setUIButtonsImages() {
  var iB = document.getElementById("infoImage");
  iB.src = infoButtonImage;
  var bB = document.getElementById("infoMAImage");
  bB.src = bulbImage;
  var mB = document.getElementById("movesImage");
  mB.src = movesImage;
  var l11 = document.getElementById("level11");
  l11.src = level11;
  var l11firstF = document.getElementById("level11firstF");
  l11firstF.src = level11firstF; 
  var l11secondF = document.getElementById("level11secondF");
  l11secondF.src = level11secondF; 
  var l12 = document.getElementById("level12");
  l12.src = level12; 
  var l12Fdash = document.getElementById("level12Fdash");
  l12Fdash.src = level12Fdash;
  var l21 = document.getElementById("level21");
  l21.src = level21;
  var l21D = document.getElementById("level21D");
  l21D.src = level21D;
  var l22 = document.getElementById("level22");
  l22.src = level22;
  var l22Ddash = document.getElementById("level22Ddash");
  l22Ddash.src = level22Ddash;
  var l22DdashSecond = document.getElementById("level22DdashSecond");
  l22DdashSecond.src = level22DdashSecond;
  var l31 = document.getElementById("level31");
  l31.src = level31;
  var l31L = document.getElementById("level31L");
  l31L.src = level31L;
  var l32 = document.getElementById("level32");
  l32.src = level32;
  var l32Rdash = document.getElementById("level32Rdash");
  l32Rdash.src = level32Rdash;
  var l41 = document.getElementById("level41");
  l41.src = level41;
  var l41R = document.getElementById("level41R");
  l41R.src = level41R;
  var l41Rdash = document.getElementById("level41Rdash");
  l41Rdash.src = level41Rdash;
  var l41U = document.getElementById("level41U");
  l41U.src = level41U;
  var l41Udash = document.getElementById("level41Udash");
  l41Udash.src = level41Udash;
  var l42 = document.getElementById("level42");
  l42.src = level42;
  var l42L = document.getElementById("level42L");
  l42L.src = level42L;
  var l42U = document.getElementById("level42U");
  l42U.src = level42U;
  var l42Ldash = document.getElementById("level42Ldash");
  l42Ldash.src = level42Ldash;
  var l42Udash = document.getElementById("level42Udash");
  l42Udash.src = level42Udash;
  var l5 = document.getElementById("level5");
  l5.src = level5;
  var l5D = document.getElementById("level5D");
  l5D.src = level5D;
  var l5R = document.getElementById("level5R");
  l5R.src = level5R;
  var l5Rdash = document.getElementById("level5Rdash");
  l5Rdash.src = level5Rdash;
  var l5Fdash = document.getElementById("level5Fdash");
  l5Fdash.src = level5Fdash;
  var l6 = document.getElementById("level6");
  l6.src = level6;
  var l6F = document.getElementById("level6F");
  l6F.src = level6F;
  var l6R = document.getElementById("level6R");
  l6R.src = level6R;
  var l6U = document.getElementById("level6U");
  l6U.src = level6U;
  var l6Udash = document.getElementById("level6Udash");
  l6Udash.src = level6Udash;
  var l7 = document.getElementById("level7");
  l7.src = level7;
  var l7B = document.getElementById("level7B");
  l7B.src = level7B;
  var l7L = document.getElementById("level7L");
  l7L.src = level7L;
  var l7U = document.getElementById("level7U");
  l7U.src = level7U;
  var l7Udash = document.getElementById("level7Udash");
  l7Udash.src = level7Udash;
  var l81 = document.getElementById("level81");
  l81.src = level81;
  var l81DFirstdash = document.getElementById("level81DFirstdash");
  l81DFirstdash.src = level81DFirstdash;
  var l81DSeconddash = document.getElementById("level81DSeconddash");
  l81DSeconddash.src = level81DSeconddash;
  var l9 = document.getElementById("level9");
  l9.src = level9;
  var l9B = document.getElementById("level9B");
  l9B.src = level9B;
  var l9L = document.getElementById("level9L");
  l9L.src = level9L;
  var l9L2 = document.getElementById("level9Lsecond");
  l9L2.src = level9secondL;
  var l9U = document.getElementById("level9U");
  l9U.src = level9U;
  var l9Udash = document.getElementById("level9Udash");
  l9Udash.src = level9Udash;
}

function infoButtonShowHide(id) {
  var x = document.getElementById(id);
  if (x.style.display === "none") {
    x.style.display = "block";
  } else {
    x.style.display = "none";
  }
}

export function enableChatbot() {
  let e = document.createElement("script"),
    t = document.head || document.getElementsByTagName("head")[0];
  (e.src = "https://cdn.jsdelivr.net/npm/rasa-webchat/lib/index.js"),
    // Replace 1.x.x with the version that you want
    (e.async = !0),
    (e.onload = () => {
      window.WebChat.default(
        {
          initPayload: "/greet",
          customData: { language: "en" },
          socketUrl: "http://localhost:5005",
          title: "Ally",
          // add other props here
        },
        null
      );
    }),
    t.insertBefore(e, t.firstChild);
  localStorage.clear();
}
