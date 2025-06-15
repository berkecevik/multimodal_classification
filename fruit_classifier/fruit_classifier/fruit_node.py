import rclpy
from rclpy.node import Node
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os
from ament_index_python.packages import get_package_share_directory

class FruitClassifierNode(Node):
    def __init__(self):
        super().__init__('fruit_classifier_node')
        self.get_logger().info("Fruit Classifier Node has started.")
        self.run()

    def run(self):
        # Use ROS 2 package path to locate CSV
        pkg_path = get_package_share_directory('fruit_classifier')
        csv_path = os.path.join(pkg_path, 'fruit_classifier', 'fruit_processed.csv')

        df = pd.read_csv(csv_path)

        # ✅ Drop rows with any missing values
        df.dropna(inplace=True)

        X = df[["Weight", "Sphericity", "Color"]]
        y = df["Label"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        self.get_logger().info("Classification Report:\n" + classification_report(y_test, y_pred))

        # Visualization
        plt.scatter(df["Weight"], df["Sphericity"], c=y, cmap='coolwarm', edgecolors='k')
        plt.xlabel("Weight")
        plt.ylabel("Sphericity")
        plt.title("Fruit Classification Visualization")
        plot_path = os.path.join(pkg_path, 'fruit_classifier', 'fruit_plot.png')
        plt.savefig(plot_path)
        self.get_logger().info(f"Saved plot as {plot_path}")


def main(args=None):
    rclpy.init(args=args)
    node = FruitClassifierNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
