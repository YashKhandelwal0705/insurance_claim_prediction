from setuptools import setup, find_packages

setup(
    name="insurance_claim_prediction",
    version="0.1.0",
    description="Machine learning solution for predicting insurance claim severity",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scikit-learn>=0.24.2',
        'matplotlib>=3.4.2',
        'seaborn>=0.11.2',
        'xgboost>=1.5.0',
        'shap>=0.40.0',
        'plotly>=5.3.0',
        'joblib>=1.0.1',
        'scipy>=1.7.0',
        'ydata-profiling>=3.6.0'
    ],
    python_requires='>=3.8',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
