# Sentiment Analysis Project

This repository contains a sentiment analysis project implemented using Python and Jupyter Notebook. The goal of the project is to classify the sentiment of movie reviews from the IMDb dataset as either positive or negative using a Logistic Regression model.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [Prediction](#prediction)
- [License](#license)

## Overview

This project demonstrates how to preprocess text data, transform it into TF-IDF features, and train a Logistic Regression model for sentiment analysis. The project also includes steps to save and load the trained model using pickle for future predictions.

## Dataset

The project uses the IMDb dataset, which contains movie reviews labeled as positive or negative. The dataset can be downloaded from [here](https://ai.stanford.edu/~amaas/data/sentiment/).

## Requirements

- Python 3.x
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Pickle

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/sentiment-analysis-project.git
    cd sentiment-analysis-project
    ```

2. Install the required packages:
    ```sh
    pip install pandas numpy scikit-learn nltk
    ```

3. Download the NLTK stopwords:
    ```sh
    python -m nltk.downloader stopwords
    ```

## Usage

1. Open the Jupyter Notebook:
    ```sh
    jupyter notebook
    ```

2. Run the cells in the notebook to preprocess the data, train the model, and make predictions.

## Project Structure

