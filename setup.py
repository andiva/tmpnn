from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='tmpnn',
    python_requires='>3.8',
    version='0.0.3',
    description='Iimplementation of TMPNN, tabular polynomial network.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(exclude=['tests', 'demo']),
    install_requires=['tensorflow', 'scipy', 'scikit-learn'],
    author='Stefan Maria Ailuro',
    author_email='ailuro.sm@gmail.com',
    url='https://github.com/andiva/tmpnn',
    test_suite='tests',
    classifiers=[
         'Programming Language :: Python :: 3',
         'License :: OSI Approved :: MIT License',
         'Operating System :: OS Independent',
     ],
)