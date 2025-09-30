import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QCheckBox, QComboBox, QSlider, QFileDialog, QPushButton, QMessageBox, QTextEdit, QStackedWidget
from PyQt5.QtCore import Qt
import pandas as pd
from typing import Dict, Union, Optional

from model_training_pmacct import run_classification, Features, MLModel

class DataAnalysisApp(QWidget):
    def __init__(self):
        super().__init__()
        self.selected_algorithm = None
        self.df = None
        self.selected_features_enums = []  # Store Features Enums
        self.initUI()
        self.results_area = None

    def initUI(self):
        self.setWindowTitle("Data Analysis Selection")
        self.setGeometry(100, 100, 800, 600)

        # --- Upload CSV File ---
        self.uploadGroup = QGroupBox("Upload your CSV file:")
        self.uploadButton = QPushButton("Choose a CSV file", self)
        self.uploadLabel = QLabel("No file selected", self)
        self.uploadButton.clicked.connect(self.showFileDialog)

        uploadLayout = QHBoxLayout()
        uploadLayout.addWidget(self.uploadButton)
        uploadLayout.addWidget(self.uploadLabel)
        self.uploadGroup.setLayout(uploadLayout)

        # --- Select Features ---
        self.featuresGroup = QGroupBox("Select Your Features:")
        self.feature_checkboxes = {}
        self.flow_options = {  # Map GUI strings to Features Enums
            "Flow Duration": Features.FLOW_DURATION,
            "Bytes": Features.BYTES,
            "Packets": Features.PACKETS,
            "Protocol": Features.PROTOCOL,
            "Source IP": Features.SRC_IP,
            "Destination IP": Features.DST_IP,
            "Source Port": Features.SRC_PORT,
            "Destination Port": Features.DST_PORT,
            "Sampling Rate": Features.SAMPLING_RATE,
            "Flows": Features.FLOWS,
            # "Timestamp Start": Features.TIMESTAMP_START,
            # "Timestamp End": Features.TIMESTAMP_END,
            "City": Features.CITY,
            "State": Features.STATE,
            "Country": Features.COUNTRY
        }
        featuresLayout = QVBoxLayout()
        h_layout1 = QHBoxLayout()
        h_layout2 = QHBoxLayout()
        h_layout3 = QHBoxLayout()
        col_count = 0
        row_count = 0
        for option_str, feature_enum in self.flow_options.items():
            checkbox = QCheckBox(option_str, self)
            self.feature_checkboxes[option_str] = checkbox
            checkbox.stateChanged.connect(self.update_feature_selection)
            if row_count == 0:
                h_layout1.addWidget(checkbox)
            elif row_count == 1:
                h_layout2.addWidget(checkbox)
            else:
                h_layout3.addWidget(checkbox)
            col_count += 1
            if col_count % 4 == 0:
                row_count +=1 
        featuresLayout.addLayout(h_layout1)
        featuresLayout.addLayout(h_layout2)
        featuresLayout.addLayout(h_layout3)
        self.featuresGroup.setLayout(featuresLayout)
        self.selected_features_count = 0

        # --- Classification Algorithm ---
        self.algorithmGroup = QGroupBox("Classification Algorithm:")
        self.knn_checkbox = QCheckBox("kNN", self)
        self.dt_checkbox = QCheckBox("Decision Tree", self)
        self.algorithmLayout = QHBoxLayout()
        self.algorithmLayout.addWidget(self.knn_checkbox)
        self.algorithmLayout.addWidget(self.dt_checkbox)
        self.algorithmGroup.setLayout(self.algorithmLayout)

        # --- Fine-Tunning ---
        self.fine_tunning = QGroupBox("Algorithm Optimization:")
        self.fine_tunning_checkbox = QCheckBox("Fine-Tuning", self)
        self.fine_tunningLayout = QHBoxLayout()
        self.fine_tunningLayout.addWidget(self.fine_tunning_checkbox)
        self.fine_tunning.setLayout(self.fine_tunningLayout)

        self.knn_checkbox.toggled.connect(self.update_algorithm_selection)
        self.dt_checkbox.toggled.connect(self.update_algorithm_selection)
        self.fine_tunning_checkbox.toggled.connect(self.update_algorithm_selection)

        # --- kNN Hyperparameters ---
        self.knnParamsGroup = QGroupBox("kNN Hyperparameters")
        self.knnParamsLayout = QVBoxLayout()

        self.weightsLabel = QLabel("Weights:", self)
        self.weightsCombo = QComboBox(self)
        self.weightsCombo.addItems(["uniform", "distance"])
        self.knnParamsLayout.addWidget(self.weightsLabel)
        self.knnParamsLayout.addWidget(self.weightsCombo)

        self.neighborsLabel = QLabel("Nearest Neighbors:", self)
        self.neighborsSlider = QSlider(Qt.Horizontal, self)
        self.neighborsSlider.setMinimum(1)
        self.neighborsSlider.setMaximum(20)
        self.neighborsSlider.setValue(5)
        self.neighborsValueLabel = QLabel(str(self.neighborsSlider.value()), self)
        self.neighborsSlider.valueChanged.connect(self.update_neighbors_label)
        self.knnParamsLayout.addWidget(self.neighborsLabel)
        self.knnParamsLayout.addWidget(self.neighborsSlider)
        self.knnParamsLayout.addWidget(self.neighborsValueLabel)

        self.algorithmLabel = QLabel("Algorithm:", self)
        self.algorithmCombo = QComboBox(self)
        self.algorithmCombo.addItems(["auto", "ball_tree", "kd_tree", "brute"])
        self.knnParamsLayout.addWidget(self.algorithmLabel)
        self.knnParamsLayout.addWidget(self.algorithmCombo)

        self.knnParamsGroup.setLayout(self.knnParamsLayout)
        self.knnParamsGroup.setVisible(False)

        # --- Decision Tree Hyperparameters ---
        self.dtParamsGroup = QGroupBox("Decision Tree Hyperparameters")
        self.dtParamsLayout = QVBoxLayout()

        self.criterionLabel = QLabel("Criterion:", self)
        self.criterionCombo = QComboBox(self)
        self.criterionCombo.addItems(["gini", "entropy"])
        self.dtParamsLayout.addWidget(self.criterionLabel)
        self.dtParamsLayout.addWidget(self.criterionCombo)

        self.splitterLabel = QLabel("Splitter:", self)
        self.splitterCombo = QComboBox(self)
        self.splitterCombo.addItems(["best", "random"])
        self.dtParamsLayout.addWidget(self.splitterLabel)
        self.dtParamsLayout.addWidget(self.splitterCombo)

        self.maxDepthLabel = QLabel("Max Depth:", self)
        self.maxDepthSlider = QSlider(Qt.Horizontal, self)
        self.maxDepthSlider.setMinimum(0)
        self.maxDepthSlider.setMaximum(50)
        self.maxDepthSlider.setValue(10)
        self.maxDepthValueLabel = QLabel(str(self.maxDepthSlider.value()), self)
        self.maxDepthSlider.valueChanged.connect(self.update_max_depth_label)
        self.dtParamsLayout.addWidget(self.maxDepthLabel)
        self.dtParamsLayout.addWidget(self.maxDepthSlider)
        self.dtParamsLayout.addWidget(self.maxDepthValueLabel)

        self.dtParamsGroup.setLayout(self.dtParamsLayout)
        self.dtParamsGroup.setVisible(False)

        # --- Hyperparameter Stacked Widget ---
        self.hyperparameterStack = QStackedWidget()
        self.hyperparameterStack.addWidget(QLabel("No algorithm selected", self))
        self.hyperparameterStack.addWidget(self.knnParamsGroup)
        self.hyperparameterStack.addWidget(self.dtParamsGroup)

        # --- Run Analysis Button ---
        self.runAnalysisButton = QPushButton("Run Analysis", self)
        self.runAnalysisButton.clicked.connect(self.run_data_analysis)

        # --- Results Display Area ---
        self.resultsGroup = QGroupBox("Analysis Results:")
        self.resultsLayout = QVBoxLayout()
        self.results_text_area = QTextEdit(self) # Use QTextEdit for multiline text
        self.resultsLayout.addWidget(self.results_text_area)
        self.resultsGroup.setLayout(self.resultsLayout)

        # --- Plot Display Area ---
        self.plotGroup = QGroupBox("Model Performance Plot:")
        self.plotLayout = QVBoxLayout()
        self.plotCanvas = None # Placeholder for the plot canvas
        self.plotGroup.setLayout(self.plotLayout)

        # --- Main Layout ---
        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.uploadGroup)
        mainLayout.addWidget(self.featuresGroup)
        mainLayout.addWidget(self.algorithmGroup)
        mainLayout.addWidget(self.fine_tunning)
        mainLayout.addWidget(self.hyperparameterStack)
        mainLayout.addWidget(self.runAnalysisButton)
        mainLayout.addWidget(self.resultsGroup)
        #mainLayout.addWidget(self.plotGroup)

        self.setLayout(mainLayout)

    def showFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "Choose a CSV file", "", "CSV files (*.csv);;All Files (*)", options=options)
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.uploadLabel.setText(f"File loaded: {file_path}")
                print(self.df.head())
            except Exception as e:
                self.uploadLabel.setText(f"Error reading file: {e}")
                QMessageBox.critical(self, "Error", f"Could not read CSV file: {e}")

    def update_feature_selection(self, state):
        checkbox = self.sender()
        feature_name = checkbox.text()
        if state == Qt.Checked:
            self.selected_features_enums.append(self.flow_options[feature_name])
        else:
            self.selected_features_enums.remove(self.flow_options[feature_name])

    def update_algorithm_selection(self):
        knn_selected = self.knn_checkbox.isChecked()
        dt_selected = self.dt_checkbox.isChecked()
        ft_selected = self.fine_tunning_checkbox.isChecked()

        if knn_selected and dt_selected:
            QMessageBox.warning(self, "Error", "kNN and Decision Tree cannot be selected at the same time.")
            self.knn_checkbox.setChecked(False)
            self.dt_checkbox.setChecked(False)
            self.fine_tunning_checkbox.setChecked(False)  # Uncheck "Find Best" as well
            self.selected_algorithm = None
            self.grid_search_enable = False
            self.hyperparameterStack.setCurrentIndex(0)  # "No algorithm selected"
            return  # Exit to prevent further processing

        if ft_selected and not knn_selected and not dt_selected:
            self.grid_search_enable = True
            self.hyperparameterStack.setCurrentIndex(0)  # "No algorithm selected"
        elif knn_selected and not dt_selected:
            self.selected_algorithm = MLModel.KNN
            self.grid_search_enable = ft_selected  # Enable grid search if "Find Best" is also checked
            self.hyperparameterStack.setCurrentIndex(1)  # kNN params
        elif dt_selected and not knn_selected:
            self.selected_algorithm = MLModel.DECISION_TREE
            self.grid_search_enable = ft_selected  # Enable grid search if "Find Best" is also checked
            self.hyperparameterStack.setCurrentIndex(2)  # DT params
        else:
            self.selected_algorithm = None
            self.grid_search_enable = False
            self.hyperparameterStack.setCurrentIndex(0)  # "No algorithm selected"

    def update_neighbors_label(self, value):
        self.neighborsValueLabel.setText(str(value))

    def update_max_depth_label(self, value):
        self.maxDepthValueLabel.setText(str(value))
    
    def get_hyperparameters(self) -> Dict[str, Union[int, str, float, None]]:
        """Collects hyperparameters based on the selected algorithm."""
        hyperparameters = {}
        if self.selected_algorithm == MLModel.KNN:
            hyperparameters["n_neighbors"] = self.neighborsSlider.value()
            hyperparameters["weights"] = self.weightsCombo.currentText()
            hyperparameters["algorithm"] = self.algorithmCombo.currentText()
        elif self.selected_algorithm == MLModel.DECISION_TREE:
            hyperparameters["criterion"] = self.criterionCombo.currentText()
            hyperparameters["splitter"] = self.splitterCombo.currentText()
            hyperparameters["max_depth"] = self.maxDepthSlider.value()
        return hyperparameters

    def run_data_analysis(self):
        if self.df is None:
            QMessageBox.warning(self, "Error", "Please upload a CSV file first.")
            return

        if not self.selected_features_enums:
            QMessageBox.warning(self, "Error", "Please select at least one feature.")
            return

        if self.selected_algorithm is None:
            QMessageBox.warning(self, "Error", "Please select a classification algorithm.")
            return

        try:
            features_with_target = [feature.value for feature in self.selected_features_enums] + ['CLASS']
            result = run_classification(
                user_df=self.df[features_with_target], 
                selected_features=self.selected_features_enums, 
                model_type=self.selected_algorithm, 
                hyperparameters=self.get_hyperparameters(),
                use_additional_data= False, 
                grid_search= self.grid_search_enable)
            
                # --- Process and Display Results ---
            if result:
                output_text = "Analysis Results:\n"
                if 'accuracy' in result:
                    output_text += f"Accuracy: {result['accuracy']:.4f}\n"
                if 'classification_report' in result:
                    output_text += f"\nClassification Report:\n{result['classification_report']}\n"
                if 'best_hyperparameters' in result:
                    output_text += f"\nBest Hyperparameters:\n{result['best_hyperparameters']}\n"

                self.results_text_area.setText(output_text)
                #self.plot_performance(result) # Call the plotting function

            else:
                self.results_text_area.setText("No results received from the backend.")

            
        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", f"An error occurred during analysis: {e}")

    
    # def plot_performance(self, results: Dict):
    #     if self.plotCanvas is not None:
    #         self.plotLayout.removeWidget(self.plotCanvas)
    #         self.plotCanvas.deleteLater()
    #         self.plotCanvas = None

    #     if 'classification_report' in results:
    #         report_str = results['classification_report']
    #         # Parse the classification report to get precision, recall, f1-score for each class
    #         lines = report_str.strip().split('\n')
    #         if len(lines) > 2:
    #             header = [h.strip() for h in lines[0].split()]
    #             data = {}
    #             for line in lines[2:]:
    #                 parts = [p.strip() for p in line.split()]
    #                 if len(parts) > 4:
    #                     class_name = parts[0]
    #                     precision = float(parts[1])
    #                     recall = float(parts[2])
    #                     f1_score = float(parts[3])
    #                     data[class_name] = {'precision': precision, 'recall': recall, 'f1-score': f1_score}

    #             if data:
    #                 classes = list(data.keys())
    #                 f1_scores = [data[c]['f1-score'] for c in classes]
    #                 fig, ax = plt.subplots()
    #                 ax.bar(classes, f1_scores, color='skyblue')
    #                 ax.set_ylabel('F1-Score')
    #                 ax.set_title('F1-Score per Class')
    #                 ax.set_ylim([0, 1.1])
    #                 fig.tight_layout()
    #                 self.plotCanvas = FigureCanvas(fig)
    #                 self.plotLayout.addWidget(self.plotCanvas)
    #                 self.plotCanvas.draw()
    #             else:
    #                 self.plot_text_in_plot_area("Could not parse classification report for plotting.")
    #         else:
    #             self.plot_text_in_plot_area("Classification report is too short to plot.")
    #     elif 'accuracy' in results:
    #         # Simple case: just display accuracy
    #         self.plot_text_in_plot_area(f"Accuracy: {results['accuracy']:.4f}")
    #     else:
    #         self.plot_text_in_plot_area("No performance metrics to plot.")

    # def plot_text_in_plot_area(self, text: str):
    #     fig, ax = plt.subplots()
    #     ax.text(0.1, 0.5, text, fontsize=12)
    #     ax.axis('off')
    #     self.plotCanvas = FigureCanvas(fig)
    #     self.plotLayout.addWidget(self.plotCanvas)
    #     self.plotCanvas.draw()

    def get_hyperparameters(self):
        """Collects hyperparameters based on the selected algorithm."""
        hyperparameters = {}
        if self.selected_algorithm == MLModel.DECISION_TREE:
            hyperparameters["criterion"] = self.criterionCombo.currentText()
            hyperparameters["splitter"] = self.splitterCombo.currentText()
            hyperparameters["max_depth"] = self.maxDepthSlider.value()
        elif self.selected_algorithm == MLModel.KNN:
            hyperparameters["n_neighbors"] = self.neighborsSlider.value()
            hyperparameters["weights"] = self.weightsCombo.currentText()
            hyperparameters["algorithm"] = self.algorithmCombo.currentText()
        return hyperparameters

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DataAnalysisApp()
    window.show()
    sys.exit(app.exec_())
