from setuptools import find_packages, setup

package_name = 'fruit_classifier'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/fruit_classifier/fruit_classifier', ['fruit_classifier/fruit_processed.csv']),
        ('share/fruit_classifier/fruit_classifier', ['fruit_classifier/iris_processed.csv']),
        ('share/fruit_classifier/fruit_classifier', ['fruit_classifier/cancer_processed.csv']),
        ('share/fruit_classifier/fruit_classifier', ['fruit_classifier/penguin_processed.csv']),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
		'fruit_node = fruit_classifier.fruit_node:main',
        'iris_node = fruit_classifier.iris_node:main',
        'svm_node = fruit_classifier.svm_node:main',
        ],
    },
)
