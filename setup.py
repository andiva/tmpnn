from setuptools import setup, find_packages
import tmpnn

setup(
    name='tmpnn',
    python_requires='>3.6',
    version=tmpnn.__version__,
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'scipy',
        'scikit-learn'],
    author='Stefan Maria Ailuro',
    author_email='steplap2003@gmail.com',
    url='https://github.com/andiva/tmpnn',
    test_suite='tests',
)