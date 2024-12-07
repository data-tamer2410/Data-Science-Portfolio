# Data Science Portfolio
Welcome to my Data Science portfolio! Here you can find my projects related to data analysis, machine learning, and data visualization.

## Projects

### [Sentiment Analysis with RNN](https://github.com/data-tamer2410/ds-sentiment-analysis-with-rnn)
**Tech Stack**:
*  **Keras**;
*  **SciKit-Learn**;
*  **Numpy**;
*  **Matplotlib**.

In this project, I developed a set of deep learning models to classify movie reviews from the IMDB dataset as positive or negative. Leveraging the sequential nature of text data, I implemented and compared multiple recurrent neural network architectures, including Simple RNN, LSTM, Bidirectional LSTM (BRNN), and Deep Bidirectional LSTM (DBRNN). The dataset contains 50,000 reviews, split evenly into training and testing subsets. Each model was trained to capture contextual information and sentiment patterns from the text, and their performance was evaluated using accuracy and loss metrics. The best results were achieved by the Bidirectional LSTM model, demonstrating its ability to handle complex text data effectively.

**Key Achievements**:
- **Advanced Architectures**: Explored and implemented multiple RNN-based architectures to improve text classification performance.  
- **Data Preprocessing**: Managed data padding and tokenization to prepare text data for recurrent models.  
- **Model Evaluation**: Visualized training and validation results, analyzing differences in model performance across architectures.  

**Why It Matters**:  
This project showcases my expertise in natural language processing, text data preparation, and applying recurrent neural networks to real-world tasks, highlighting my ability to build, optimize, and evaluate deep learning models for practical applications.  

---

### [Eyes Classification](https://github.com/data-tamer2410/ds-eyes-classification)
**Tech Stack**:
*  **Keras**;
*  **SciKit-Learn**;
*  **Numpy**;
*  **Matplotlib**.

In this project, I developed a deep learning model for classifying images of eyes into different categories (open or closed). The dataset consists of various images of human eyes captured in real-life conditions. The model was built using convolutional neural networks (CNNs) and trained to recognize subtle features of the eye, providing a robust classification solution that can be used in applications such as facial recognition or eye-tracking systems.

**Key Achievements**:
- **Deep Learning Architecture**: Implemented CNNs for efficient feature extraction and classification of images.
- **Data Preprocessing**: Applied data augmentation techniques to enhance the model's generalization ability.
- **Real-World Use Cases**: The model can be used for real-time applications such as eye health monitoring, security, and accessibility technologies.

**Why It Matters**:  
This project demonstrates my expertise in image classification, computer vision, and deep learning techniques, showcasing my ability to apply neural networks to practical real-world problems. It highlights my skills in handling image data, training complex models, and optimizing them for performance.

---

### [Fashion MNIST Classifier 2.0](https://github.com/data-tamer2410/ds-fashion-mnist-classifier-2.0)
**Tech Stack**:
*  **Keras**;
*  **SciKit-Learn**;
*  **Numpy**;
*  **Matplotlib**;
*  **Pandas**;
*  **Streamlit**;
*  **Docker**.

This project explores advanced techniques for image classification using the Fashion MNIST dataset, where I implemented multiple neural network architectures to enhance accuracy. The task involved classifying images of various clothing items based on grayscale data, using methods such as convolutional neural networks (CNN), VGG16 for feature extraction, and fine-tuning pre-trained models for better performance. As part of this project, I developed an interactive web dashboard to visualize model predictions and provide insights into the dataset [Dashboard](https://ds-fashion-mnist-classifier-2-0.onrender.com). 

**Key Achievements**:
- **Multiple Model Approaches:** Developed custom CNN, utilized VGG16 for feature extraction, and fine-tuned VGG16 layers to maximize classification accuracy.  
- **Previous Work Integration:** Built on a previous project where I created a simpler fully connected classifier for the Fashion MNIST dataset.  
- **Real-World Impact:** Demonstrated the potential for using these techniques in practical applications like retail and fashion e-commerce.  

**Why It Matters**:  
This project demonstrates my ability to work with cutting-edge deep learning techniques, fine-tuning models for optimal performance, and integrating past knowledge to create more advanced solutions. It showcases my expertise in image classification and my capacity to develop solutions directly applicable to industries such as fashion and retail.  

---

### [Fashion MNIST Classifier](https://github.com/data-tamer2410/ds-fashion-mnist-classifier)
**Tech Stack**:
*  **TensorFlow**;
*  **Keras**;
*  **SciKit-Learn**;
*  **Numpy**;
*  **Matplotlib**;
*  **Optuna**.

This project presents a robust image classification model for the Fashion MNIST dataset, where I applied a combination of neural network design and hyperparameter optimization to achieve high accuracy. The model classifies various clothing types based on grayscale images, simulating real-world applications in retail, e-commerce, and fashion industries. To maximize performance, I integrated Optuna for hyperparameter tuning, optimizing parameters such as the number of layers, activation functions, batch size, and optimizer types. The model achieved **93.24% accuracy on training data** and **88.3% accuracy on test data**, surpassing the project’s original accuracy goal. 

**Key Achievements**:
- **High Accuracy and Stability:** Model validation accuracy of 89.44% with low loss, ensuring reliable performance in test scenarios.
- **Data-Driven Hyperparameter Optimization:** Applied Optuna to refine model parameters across 50 trials, ensuring the most effective combination of settings.
- **Real-World Relevance:** The techniques and outcomes of this project are directly applicable to any business or application requiring fast, accurate image-based classification.

**Why It Matters**:  
This project not only demonstrates expertise in deep learning model building but also in systematic optimization—a crucial skill in data science roles that require balancing model performance with computational efficiency. This ability to deliver optimized, high-performance models showcases my readiness for real-world machine learning challenges in industry applications.

---

### [Handwritten Digit Recognition](https://github.com/data-tamer2410/ds-handwritten-digit-recognition)
**Tech Stack**:
*  **TensorFlow**;
*  **Keras**;
*  **SciKit-Learn**;
*  **Numpy**;
*  **Matplotlib**.

In this project, I developed a neural network model to accurately recognize handwritten digits from the MNIST dataset. The architecture features two hidden layers with sigmoid activation functions and a softmax output layer for multi-class classification. 

**Key Achievements**:
- **High Accuracy**: Achieved 92.40% accuracy on training data and approximately 89.87% in cross-validation.
- **Data Processing**: Implemented effective data normalization and transformation techniques to optimize model performance.
- **Evaluation Metrics**: Used precision, F1-score, and ROC-AUC to assess model effectiveness, demonstrating its ability to reliably distinguish between digits.

**Why It Matters**: 
This project showcases my proficiency in deep learning, data preprocessing, and model evaluation, highlighting my ability to tackle real-world problems in computer vision.

---

### [Accelerometer Activity Classification](https://github.com/data-tamer2410/ds-accelerometer-activity-classification)
**Tech Stack**:
*  **SciKit-Learn**;
*  **Optuna**;
*  **Pandas**;
*  **Numpy**;
*  **Matplotlib**;
*  **Seaborn**.

In this project, I developed a human activity recognition system that accurately classifies activities such as idle, running, walking, and climbing stairs using accelerometer data. Leveraging machine learning algorithms, including k-NN and Random Forest, I implemented a pipeline for data preprocessing, feature extraction, and model evaluation. The project demonstrates my ability to apply advanced analytics to real-world problems and highlights my skills in data science, machine learning, and software development. The high accuracy achieved not only showcases the effectiveness of the models used but also emphasizes the potential for real-time applications in health monitoring and smart device integration.

---

### [MovieLens Recommender System](https://github.com/data-tamer2410/ds-movielens-recommender-system)
**Tech Stack**:
*  **Surprise**;
*  **Optuna**;
*  **Pandas**;
*  **Numpy**;
*  **Scipy**;
*  **Matplotlib**.

Developed a recommendation system leveraging collaborative filtering with the Surprise library, delivering accurate, data-driven user preferences for personalized content. This project demonstrates my proficiency in machine learning and recommendation algorithms, including data processing, model training, and performance optimization, tailored to enhance user experience through intelligent recommendations.

---

### [House Price Predictor](https://github.com/data-tamer2410/ds-house-price-predictor)
**Tech Stack**:
*  **SciKit-Learn**;
*  **Optuna**;
*  **Pandas**;
*  **Numpy**;
*  **Matplotlib**;
*  **Seaborn**.

In this project, I developed a comprehensive suite of regression models to predict outcomes using diverse machine learning algorithms, including Linear Regression, Lasso Regression, Random Forest, and Gradient Boosting. I employed techniques such as cross-validation and hyperparameter tuning using Optuna to optimize model performance. 
The project involved detailed data preprocessing, feature selection, and rigorous evaluation metrics, including RMSE, MAE, and R² scores, ensuring robust model validation. Visualizations were created to illustrate model predictions versus actual outcomes, highlighting the efficacy of each algorithm. 
This project showcases my skills in data analysis, feature engineering, and machine learning, demonstrating my ability to drive insights and make data-driven decisions.

---

### [Cluster Insight](https://github.com/data-tamer2410/ds-cluster-insight)
**Tech Stack**:
*  **SciKit-Learn**;
*  **Numpy**.
*  **Pandas**;
*  **Matplotlib**;
*  **Seaborn**.

Developed a robust KMeans clustering model to segment data into optimal groups, achieving high accuracy by leveraging the elbow method for cluster selection. The project includes clear visualizations comparing predicted clusters with actual targets, highlighting model strengths and limitations in complex data intersections. This practical clustering solution demonstrates proficiency in data preprocessing, model evaluation, and insightful visual analysis, showcasing essential skills for data-driven decision-making.
  
---

### [Developer Salary Analysis](https://github.com/data-tamer2410/ds-developer-salary-analysis)
**Tech Stack**:
*  **Pandas**;
*  **Numpy**;
*  **Matplotlib**;
*  **Seaborn**.

In this project, I conducted a comprehensive analysis of developer salaries using a dataset from a 2017 developer survey. I performed data cleaning and preprocessing, focusing on key features such as job roles, programming languages, experience, and salary changes. Utilizing Python libraries like Pandas, Seaborn, and Matplotlib, I visualized relationships between salaries and factors like English proficiency, age, and company size.
The analysis revealed significant insights, including the correlation between work experience, language skills, and salary levels. I generated descriptive statistics and visualizations that illustrated how salaries vary by job title and company size. Additionally, I highlighted the trends in the developer workforce's age distribution and the impact of education on salary. This project demonstrates my ability to derive actionable insights from data and communicate findings effectively.

---

### [Text Summarization](https://github.com/data-tamer2410/ds-text-summarization)
**Tech Stack**:
*  **NLTK**;
*  **LangDetect**;
*  **Heapq**;
*  **String**.

This project demonstrates the ability to automatically generate concise summaries of long texts using Natural Language Processing (NLP) with NLTK. The process involves text preprocessing (tokenization, stopword removal, and punctuation handling), followed by frequency analysis of key words. The project calculates sentence scores based on the importance of contained words and generates a summary by selecting the highest-scoring sentences. This method can be useful for tasks such as document summarization, information retrieval, and content extraction, and showcases proficiency in NLP techniques.

---

### [Analysis of Birth Rates in Ukrainian Regions](https://github.com/data-tamer2410/ds-analysis-of-birth-rates-in-ukrainian-regions)
**Tech Stack**:
*  **Pandas**;
*  **Numpy**;
*  **Matplotlib**;
*  **Seaborn**.

In this project, I conducted a comprehensive analysis of fertility rates across various regions in Ukraine over a span of nearly seven decades. Utilizing data scraped from Wikipedia, I meticulously cleaned and transformed the dataset, which includes regional birth rates from 1950 to 2019, into a structured format for analysis.

**Key insights**:
- The lowest fertility rates were observed in 2000, highlighting a demographic crisis at the turn of the millennium.
- A noticeable recovery in birth rates began in 2000, suggesting a gradual demographic rebound.
- The dramatic decline in the Luhansk region's birth rate in 2014 correlates with the onset of military conflict, providing context to the data.
- By 2019, regions like Kyiv and Zakarpattia exhibited above-average fertility rates, while Sumy had the lowest.

**Why It Matters**: 
The project utilized Python libraries such as Pandas, NumPy, and Seaborn for data manipulation and visualization, presenting findings through informative graphs that illustrate the shifting dynamics of fertility rates over time. This analysis not only enhances understanding of demographic trends but also emphasizes the impact of socio-political events on population metrics.

---

### [Manipulation with SQLite](https://github.com/data-tamer2410/ds-manipulation-with-sqlite)
**Tech Stack**:
*  **SQLite**;
*  **SQL**.

In this project, I demonstrated my knowledge of the SQL programming language and my ability to work with SQLite in Python.

---

### [Manipulation with MongoDB](https://github.com/data-tamer2410/ds-manipulation-with-mongodb)
**Tech Stack**:
*  **MongoDB**;
*  **PyMongo**;
*  **BeautifulSoup**;
*  **Requests**;
*  **JSON**.

This repository showcases my MongoDB skills using pymongo, and scraping and parsing skills using beautifulsoup4 and requests.

---

### [Console Bot Helper](https://github.com/data-tamer2410/ds-console-bot-helper)
**Tech Stack**:
*  **Docker**;
*  **DateTime**.

This is a console bot assistant, it is not related to data science, with this project I want to show the ability to work with Docker.
