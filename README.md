Allure Front-end Documentation

Installation:

Some commands might differ depending on what operating system you are running the project on. These commands are primarily for windows.

Creating Rasa Environment

Install Anaconda and open Anaconda Prompt

Create a separate environment in Anaconda with any Python version from 3.6-3.8

    “conda create -n rasa python=3.6” (where “rasa” is the name of the environment)

Activate your rasa environment using “conda activate rasa”

Then “pip install rasa==2.6.2” to install rasa.

Also run “pip install python-firebase”

Finally install pickle using “pip install pickle5"

Running Rasa Environment

Once in your project ’s root folder. Cd inside chatbot folder.

Make sure rasa environment is activated otherwise run “conda activate rasa”

To train the model “rasa train”

Run the model using “rasa run -m models --enable-api --cors "\*"”

Along with that run this command “rasa run actions” in a separate terminal

Installing and Running Project

Make sure you have “nodejs” with “npm” installed.

Then cd into the project ’s main folder again and run “npm i”

Then to run the local project “npm start”
