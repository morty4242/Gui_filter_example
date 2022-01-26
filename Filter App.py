import numpy as np
import pickle
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtWidgets, QtGui
from filter_app_window import Ui_MainWindow
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.pyplot import Figure


class MainApp(Ui_MainWindow):
    def __init__(self, parent):
        self.setupUi(parent)
        parent.setWindowTitle('Interactive Filter')

        # generate data
        self.X, self.x_inliers, self.x_outliers = self.generate_data()

        # plot data
        self.plot_generated_data()

        # connect button with function
        self.LocalOutlierButton.clicked.connect(lambda: self.apply_filter('Local Outlier Factor'))
        self.RobustCovButton.clicked.connect(lambda: self.apply_filter('Robust covariance'))
        self.OneClassSVMButton.clicked.connect(lambda: self.apply_filter('One-Class SVM'))
        self.IsolationForestButton.clicked.connect(lambda: self.apply_filter('Isolation Forest'))
        self.SaveButton.clicked.connect(self.save_prediction)

        # set up filters
        self.filter_methods_dict = {'Robust covariance': EllipticEnvelope(contamination=0.03),
                                    'One-Class SVM': svm.OneClassSVM(nu=0.03, kernel='rbf', gamma=0.1),
                                    'Isolation Forest': IsolationForest(contamination=0.03, random_state=42),
                                    'Local Outlier Factor': LocalOutlierFactor(n_neighbors=35, contamination=0.03)}

    @staticmethod
    def generate_data():
        np.random.seed(17)

        # Generate train data
        x_inliers = 0.3 * np.random.randn(1000, 2)
        x_inliers = np.r_[x_inliers + 2, x_inliers - 2]

        # Generate some outliers
        x_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
        x = np.r_[x_inliers, x_outliers]

        return x, x_inliers, x_outliers

    def plot_generated_data(self):
        graph_object = GraphWidget()
        layout = QtWidgets.QVBoxLayout(graph_object)
        figure = Figure(figsize=(5, 5), dpi=100)
        graph_object.canvas = FigureCanvasQTAgg(figure)
        toolbar = NavigationToolbar(graph_object.canvas, graph_object)
        layout.addWidget(toolbar)
        layout.addWidget(graph_object.canvas)
        graph_object.setLayout(layout)
        graph_object.scatter_plot(self.x_inliers, self.x_outliers)
        self.DataFrame.setLayout(layout)

    def apply_filter(self, selected_filter):
        clf = self.filter_methods_dict[selected_filter]
        if selected_filter == 'Local Outlier Factor':
            y_pred = clf.fit_predict(self.X)
        else:
            y_pred = clf.fit(self.X).predict(self.X)

        plt.title('{}'.format(selected_filter))
        plt.scatter(self.X[:, 0], self.X[:, 1], color='r', s=3., label='Data points')
        for i in range(len(y_pred)):
            if y_pred[i] == -1:
                plt.scatter(self.X[i, 0], self.X[i, 1], s=30, marker='*', edgecolors='y')

        plt.tight_layout()
        plt.legend(loc='upper left')
        plt.show()

        self.y_pred = y_pred

    def save_prediction(self):
        filename = 'prediction_result'
        outfile = open(filename, 'wb')
        pickle.dump(self.y_pred, outfile)
        outfile.close()


class GraphWidget(QtWidgets.QWidget):
    def __init__(self, parent_ =None):
        super().__init__(parent_)
        self.setGeometry(QtCore.QRect(10, 60, 1000, 530))
        self.setObjectName('graph_widget')
        self.canvas = None

    def setup_figures(self):
        layout = QtWidgets.QVBoxLayout(self)
        figure = Figure(figsize=(5, 5), dpi=100)
        self.canvas = FigureCanvasQTAgg(figure)
        toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def scatter_plot(self, X_inliers, X_outliers):
        self.canvas.ax = self.canvas.figure.subplots()
        self.canvas.ax.clear()
        self.canvas.ax.scatter(X_inliers[:, 0], X_inliers[:, 1], color='r', s=3., label='Normal points')
        self.canvas.ax.scatter(X_outliers[:, 0], X_outliers[:, 1], color='b', s=4., label='Outlier points')


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = MainApp(MainWindow)
    MainWindow.show()

    sys.exit(app.exec_())