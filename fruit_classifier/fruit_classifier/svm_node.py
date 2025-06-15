import rclpy
from rclpy.node import Node
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
from ament_index_python.packages import get_package_share_directory
import matplotlib.pyplot as plt

class SVMClassifierNode(Node):
    def __init__(self):
        super().__init__('svm_classifier_node')
        self.get_logger().info("SVM Classifier Node has started.")
        self.run()

    def run(self):
        pkg_path = get_package_share_directory('fruit_classifier')
        self.train_and_evaluate(
            os.path.join(pkg_path, 'fruit_classifier', 'cancer_processed.csv'),
            "diagnosis", "Breast Cancer"
        )
        self.train_and_evaluate(
            os.path.join(pkg_path, 'fruit_classifier', 'penguin_processed.csv'),
            "species", "Penguin"
        )

    def train_and_evaluate(self, file_path, label_column, dataset_name):
        df = pd.read_csv(file_path)
        df.dropna(inplace=True)

        X = df.drop(columns=[label_column])
        y = df[label_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = SVC(kernel='linear')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        report = classification_report(y_test, y_pred)
        self.get_logger().info(f"\n===== {dataset_name} Dataset =====\n{report}")

        # ---- PLOTTING SECTION ----
        import matplotlib.pyplot as plt

        if dataset_name == "Breast Cancer":
            if "radius_mean" in df.columns and "texture_mean" in df.columns:
                x_feature = "radius_mean"
                y_feature = "texture_mean"
            else:
                x_feature = X.columns[0]
                y_feature = X.columns[1]
        else:  # Penguin
            if "flipper_length_mm" in df.columns and "body_mass_g" in df.columns:
                x_feature = "flipper_length_mm"
                y_feature = "body_mass_g"
            else:
                x_feature = X.columns[0]
                y_feature = X.columns[1]

        plt.figure()
        plt.scatter(df[x_feature], df[y_feature], c=y, cmap='viridis', edgecolors='k')
        plt.xlabel(x_feature)
        plt.ylabel(y_feature)
        plt.title(f"{dataset_name}: {x_feature} vs {y_feature}")

        # Save plot
        plot_path = os.path.join(os.path.dirname(file_path), f"{dataset_name.lower().replace(' ', '_')}_plot.png")
        plt.savefig(plot_path)
        self.get_logger().info(f"Saved plot to: {plot_path}")


def main(args=None):
    rclpy.init(args=args)
    node = SVMClassifierNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
