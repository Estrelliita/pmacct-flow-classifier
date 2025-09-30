# PMACCT Flow Application Classification GUI
## 📝 Description

The PMACCT Flow Application Classification GUI is a desktop application designed to streamline the process of training and testing machine learning classifiers on network flow data collected using pmacct.

This tool provides an intuitive Python/Qt GUI that allows users to easily upload and combine multiple flow data CSV files, perform necessary data pre-processing (like IP lookup and flow duration calculation), select specific flow features, and train a classification model. The target variable for the classification models is the application label (CLASS) provided by nDPI (Network Deep Packet Inspection).


## ✨ Key Features

* CSV Data Management: Upload and automatically combine multiple CSV files containing network flow data into a single training/testing dataset.

* Dynamic Feature Engineering: IP Geographic Lookup: Calculate and add IP city, state, and country information to the dataset using the ipinfo.io API if the data is missing.

* Flow Duration Calculation: Automatically calculate the flow duration (in seconds) from the TIMESTAMP_START and TIMESTAMP_END columns using Pandas.

* Feature Selection: Select desired flow features (e.g., Bytes, Packets, Flow Duration, Source/Destination IP/Port, City, State, Country) via checkboxes for model training.

* Classification Models: Choose between two powerful classification algorithms for network flow classification:

* k-Nearest Neighbors (kNN)

* Decision Tree

* Algorithm Optimization (Fine-Tuning): An optional feature that runs a series of tests using RandomizedSearchCV to find the best hyperparameters for the selected model to maximize accuracy.

* Manual Hyperparameter Control: Manually adjust key parameters for both kNN (weights, neighbors, algorithm) and Decision Tree (criterion, splitter, max depth) via sliders and drop-down menus.

* Results Reporting: Run the analysis with a single click and display the model's Accuracy and detailed Classification Report in the GUI.


## 🛠️ Technology Stack
| Category |	Tool / Library	| Purpose |
| :------- | :--------------: | -------: |
|Language	 | Python           | Core development language. |
| GUI Framework	| PyQt5/Qt for Python | Building the graphical user interface. |
| Data Processing	| Pandas | DataFrame manipulation, CSV combining, and flow duration calculation. |
| Machine Learning | Scikit-learn | Implementing kNN and Decision Tree models, data scaling, and hyperparameter tuning (RandomizedSearchCV). |
| Data Source	| pmacct |Used for initial network flow data collection (CSV files). |
| External API | ipinfo.io | Used for IP geographic lookups (City, State, Country).| Export to Sheets |

## 🚀 Getting Started
These instructions will get you a copy of the project up and running on your local machine.

Prerequisites
You'll need the following installed:

Python 3.x

pip (Python package installer)


curl (required for the IP geographic lookup subprocess) 

Installation
Clone the repository:

Bash

git clone [Your-Repository-URL]
cd pmacct-flow-classifier-gui
Install the required Python packages:
The project relies on libraries like PyQt5, pandas, and scikit-learn. Install them using pip:


Bash

pip install -r requirements.txt
(Note: You will need to create a requirements.txt file listing all dependencies.)

Set up IP Info Token (Optional):
For more than 1,000 monthly IP lookups, you should sign up for a token on 

ipinfo.io and update the ipinfo_TOKEN variable in combine_csv.py (or gui.py).


Run the application:
Launch the GUI from your terminal:

Bash

python gui.py
⚙️ Usage Workflow

Upload Data: Click "Choose a CSV file" to select one or more flow data CSV files.



Preprocessing (Optional): The application will offer to calculate Flow Duration or perform IP Lookups if the required columns (timestamps or IP addresses) are present but the target features are missing.



Select Features: Use the checkboxes under Select Your Features to choose the flow features you want to use to train the classifier.


Select Algorithm: Check the box for either kNN or Decision Tree.

Set Hyperparameters:

Check 

"Fine-Tuning" to automatically find the best parameters.


OR manually adjust the parameters (e.g., Nearest Neighbors, Max Depth) using the provided sliders and drop-down menus.


* Run Analysis: Click "Run Analysis" to start the training and testing process.
* View Results: The Analysis Results box will display the model's Accuracy and a detailed Classification Report.

## 👤 Authors
Daniel Flores

Estrella Lara

Abigail Nevarez
