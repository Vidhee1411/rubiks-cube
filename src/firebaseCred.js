// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyBJNo2rP0u5TAV2nHqvDMo2XpjwBhK0zuw",
  authDomain: "allure-chatbot.firebaseapp.com",
  databaseURL: "https://allure-chatbot-default-rtdb.firebaseio.com",
  projectId: "allure-chatbot",
  storageBucket: "allure-chatbot.appspot.com",
  messagingSenderId: "114600761213",
  appId: "1:114600761213:web:995f39d583c986878c9e14",
};

// Initialize Firebase
export const app = initializeApp(firebaseConfig);
