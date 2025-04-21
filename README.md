# Python_Career-Recommendation-System
The Career Recommendation System developed in this project has demonstrated its capability to provide accurate and relevant career suggestions based on user inputs regarding their skills and interests. Through the application of TF-IDF vectorization, dimensionality reduction using Truncated SVD, and Nearest Neighbors classification, the model successfully recommends suitable careers, outperforming a baseline model in key performance metrics.

Workflow of the Project:
The career recommendation system follows a structured workflow that involves data preprocessing, model training, user input handling, and result presentation. The workflow is divided into the following stages:
Step 1: Data Collection and Preprocessing
•	The dataset used for the career recommendation system is a CSV file containing users' skills, interests, and career titles.
•	The dataset undergoes cleaning to remove missing values, duplicate entries, and inconsistencies.
•	A new feature, combined_text, is created by merging the "Interests" and "Skills" columns to serve as input for text processing.
Step 2: Text Processing using TF-IDF Vectorization
•	The TfidfVectorizer from sklearn.feature_extraction.text is used to transform textual data into numerical representations.
•	Stop words are removed to improve efficiency.
•	A maximum of 1000 features is selected to reduce complexity.
Step 3: Dimensionality Reduction using Truncated SVD
•	Since TF-IDF vectors are high-dimensional, TruncatedSVD (Singular Value Decomposition) is applied to reduce dimensionality to 100 components while retaining relevant information.
•	This step enhances computation speed and avoids overfitting.
Step 4: Standardization of Features
•	The reduced feature matrix is scaled using StandardScaler to bring all values into a standard range.
•	This step ensures that the distance-based recommendation model works efficiently.
Step 5: Career Recommendation using Nearest Neighbors (KNN)
•	A NearestNeighbors model with cosine similarity is used to find the top 5 careers that match the user’s skills and interests.
•	Cosine similarity measures the angle between vectors, making it effective for text-based recommendations.
Step 6: GUI Implementation with Tkinter
•	A Graphical User Interface (GUI) is built using tkinter to provide user-friendly interaction.
•	Users can select skills and interests from dropdown menus.
•	Clicking the "Get Career Recommendation" button processes the input and displays the top 5 career suggestions.
