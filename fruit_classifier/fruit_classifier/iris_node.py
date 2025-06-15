import rclpy
from rclpy.node import Node
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import os
from ament_index_python.packages import get_package_share_directory

class IrisClassifierNode(Node):
    def __init__(self):
        super().__init__('iris_classifier_node')
        self.get_logger().info("Iris Classifier Node has started.")
        self.run()

    def run(self):
        # Load dataset from installed share path
        pkg_path = get_package_share_directory('fruit_classifier')
        csv_path = os.path.join(pkg_path, 'fruit_classifier', 'iris_processed.csv')

        df = pd.read_csv(csv_path)
        df.dropna(inplace=True)

        X = df.drop(columns=["Species"])
        y = df["Species"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        classifiers = {
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier()
        }

        for name, model in classifiers.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred)
            self.get_logger().info(f"\n{name} Classification Report:\n{report}")

        # Plot PetalLength vs PetalWidth
        plt.figure(figsize=(6, 4))
        plt.scatter(df["PetalLengthCm"], df["PetalWidthCm"], c=y, cmap='viridis', edgecolors='k')
        plt.xlabel("Petal Length (scaled)")
        plt.ylabel("Petal Width (scaled)")
        plt.title("Iris Dataset - PetalLength vs PetalWidth")
        plot_path = os.path.join(pkg_path, 'fruit_classifier', 'iris_plot.png')
        plt.savefig(plot_path)
        self.get_logger().info(f"Saved iris plot at {plot_path}")

def main(args=None):
    rclpy.init(args=args)
    node = IrisClassifierNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
