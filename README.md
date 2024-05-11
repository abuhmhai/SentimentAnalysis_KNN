
# Product Review Classification using Machine Learning

## Introduction
This project aims to classify product reviews as positive or negative using machine learning algorithms. The classification is based on the sentiment expressed in the review text. We will employ various text vectorization techniques and machine learning models to achieve this classification.

## Approach
The project follows these main steps:

1. **Data Collection**: Gather product reviews from online shopping websites. Each review consists of text and a corresponding rating indicating user sentiment.

2. **Data Preprocessing**:
   - Remove HTML tags and punctuation from the review text.
   - Filter out unnecessary elements and clean the text data.
   - Ensure the correctness of the helpfulness ratio (numerator should be less than or equal to the denominator).
   - Deduplicate the data based on user ID, profile name, time, and text.

3. **Text Vectorization**:
   - Use Bag of Words (BOW) and TF-IDF to convert text data into numerical vectors.
   - Utilize Word2Vec and TF-IDF Weighted Word2Vec for embedding-based vectorization.
   
4. **Classification**:
   - Apply the K-nearest Neighbors (KNN) algorithm for classification.
   - Evaluate the performance of the classification model using accuracy, precision, recall, and F1-score metrics.
   
5. **Model Evaluation**:
   - Analyze the confusion matrix to understand the model's performance.
   - Compute precision, recall, and F1-score for both positive and negative classes.
   
6. **Results Analysis**:
   - Interpret the model's accuracy and performance metrics.
   - Discuss strengths and weaknesses of the classification model.

## Installation
1. Install Python (version 3.x).
2. Install the required libraries by running the following command in the terminal or command prompt:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Run the `Classification_Project.ipynb` Jupyter notebook to see the complete workflow.
2. Customize the code for your specific use case or dataset.
3. Experiment with different text vectorization techniques and machine learning algorithms.

## Project Structure
- `Classification_Project.ipynb`: Jupyter notebook containing source code and detailed explanations.
- `data/`: Directory containing input data.
- `README.md`: Project overview and usage guide documentation.

## Technology Used
- Programming Language: Python
- Main Libraries: pandas, numpy, scikit-learn, nltk, gensim, matplotlib, seaborn

## Author
- BewxSevez

## License
This project is released under the [MIT License](https://opensource.org/licenses/MIT).

