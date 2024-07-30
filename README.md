# system_grading_islamic_questions

I.	Introducion:

In the ever-evolving landscape of education, the integration of technology has become imperative to enhance learning experiences. Our project, the "Arabic Automated Short Answers Grading System and Smart Assistance for Islamic Education for Schoolers," is a groundbreaking initiative designed to revolutionize the assessment and support mechanisms in Arabic-language education.
The core objective of this system is to provide precise and efficient grading for students based on their responses. Tailored for the Arabic language, our system incorporates a comprehensive dataset meticulously curated from diverse sources, including Arabic websites, datasets, books, and more.
To accomplish this ambitious goal, our project encompasses several key components:
1.	Data Collection:
Scraping data from a multitude of sources, ensuring a rich and varied dataset that accurately reflects the linguistic nuances of Arabic.
2.	Natural Language Processing (NLP) Pipeline:
Establishment of a robust Arabic NLP pipeline to process and analyze the collected data, laying the foundation for subsequent stages.
3.	Exploratory Data Analysis (EDA):
Employing various EDA techniques to gain insights into the dataset, ensuring a thorough understanding of the linguistic patterns and nuances inherent in Arabic responses.
4.	Word Embedding and Encoding:
Implementation of advanced techniques such as word embedding and encoding to represent and understand the semantic relationships within the Arabic language.
5.	Model Training:
Leveraging cutting-edge models including Recurrent Neural Networks (RNN), Long Short-Term Memory networks (LSTM), and Transformers to train the system for accurate grading.
6.	Text Generation for Student Assistance:
Utilizing state-of-the-art models such as BERT (Fine Tuning), and fine-tuning Language Models (LLMs) like GPT-3, Falcon, and Alpaca for text generation to provide intelligent assistance to students.
7.	Model Evaluation:
Rigorous evaluation of models using diverse metrics including ROC, accuracy, F1 score, BLEU score, among others, to identify and select the best-performing model.
8.	Model Deployment:
Deploying the chosen model using Docker containers and Kubernetes orchestration, ensuring scalability and efficiency. The system will be accessible through a user-friendly SAP web application built with Angular.
By combining advanced technologies and methodologies, our project aims to contribute significantly to the improvement of Arabic-language education, offering an automated grading system and intelligent support that aligns with the unique characteristics of the language.

II.	Scrapping :

In the initial phase of our project, the imperative task of data collection was undertaken. Recognizing the significance of diverse and comprehensive data sources, we engaged in scraping information from a multitude of platforms. Our data collection efforts encompassed Websites, PDFs, databases, CSV files, Audio files, YouTube videos, and various other repositories.

![image](https://github.com/Ayoubelfallah/system_grading_islamic_questions/assets/93795377/645de944-6e80-4c9c-97cc-2488e330dc9a)


On the Scrapping.ipynb file, you’ll get to the details of the code.
The primary objective was to assemble a diverse set of questions that would serve as the foundation for our deep learning system. Simultaneously, obtaining corresponding answers for these questions was crucial for the training of our models. To streamline this process, a meticulous approach was adopted, involving the creation of a structured Google form to systematically gather answers.
We managed to gather 107 answers from the following Form:

![image](https://github.com/Ayoubelfallah/system_grading_islamic_questions/assets/93795377/8132d5b0-6120-4ecd-9faa-df9801084ba5)

![image](https://github.com/Ayoubelfallah/system_grading_islamic_questions/assets/93795377/eb1a4487-bb3d-4ad2-baea-673c7b2e16ca)


We stored our database as follows:

![image](https://github.com/Ayoubelfallah/system_grading_islamic_questions/assets/93795377/a29a55eb-3da8-405d-b2cf-ab9b557f1617)

The form questions are obtained through the use of scraping:

![image](https://github.com/Ayoubelfallah/system_grading_islamic_questions/assets/93795377/62f785fd-fc48-49a0-88fd-e631d6abf0bb)

III.	EDA:
EDA (Exploratory Data Analysis) involves analyzing and visualizing data to gain insights and understand its underlying patterns.

![image](https://github.com/Ayoubelfallah/system_grading_islamic_questions/assets/93795377/e0c82a15-e822-4883-9876-70eaec946cea)

![image](https://github.com/Ayoubelfallah/system_grading_islamic_questions/assets/93795377/c2a3bdd7-4338-46a3-8027-59a0211ab062)

![image](https://github.com/Ayoubelfallah/system_grading_islamic_questions/assets/93795377/ab7a7c20-c64b-4935-8209-8c50cd49e2ff)


IV.	Data pre-preprocessing:

Data pre-preprocessing is the initial stage of data preparation before standard preprocessing steps. It involves tasks such as data collection, cleaning, and early transformations to enhance data quality for subsequent analyses. This phase aims to address immediate issues and lay the groundwork for more extensive preprocessing.
In our data processing pipeline, we utilized the nltk.tokenize module's word_tokenize function for tokenization , This function allowed us to break down sentences or paragraphs into individual words or tokens. By using word_tokenize, we obtained a list of tokens, where each token represented a distinct word in the text.
We utilized also the gensim.models module's Word2Vec class for word embedding generation.

![image](https://github.com/Ayoubelfallah/system_grading_islamic_questions/assets/93795377/c81cf36c-8fa0-4f6b-b44e-9ce338e48096)

Tokenization :

tokenization refers to the process of breaking down a sequence of text into individual units, known as tokens.

![image](https://github.com/Ayoubelfallah/system_grading_islamic_questions/assets/93795377/eba5b740-ab27-4697-b2dc-a23f0493f2c1)


Lemmatization :

Lemmatization is valuable in text preprocessing for various NLP applications as it helps reduce the dimensionality of the vocabulary and ensures that different inflections of a word are treated as the same term.

![image](https://github.com/Ayoubelfallah/system_grading_islamic_questions/assets/93795377/50033b18-4307-4d6a-b348-57dfc9b29ed6)

Split Data :
Data splitting is the practice of dividing a dataset into training and testing sets for machine learning model evaluation.

![image](https://github.com/Ayoubelfallah/system_grading_islamic_questions/assets/93795377/5b19d6ea-6861-4f7e-9bc4-523e7601c06b)

V.	The model building process :

We thought it would be useful to use Machine learning hand by hand to deep learning to see which will work the best, and for all questions we got a higher accuracy when it comes to LSTM model!

KNN:

The K-Nearest Neighbors (KNN) model is implemented in the provided code snippet as a function. KNN is a versatile and intuitive classification algorithm that makes predictions based on the majority class of the k-nearest neighbors in the feature space. In this case, the KNeighborsClassifier is configured with n_neighbors=6, meaning it considers the six nearest neighbors when making predictions. The model is trained using the fit method on the training data (X_train for features and Y_train for target labels). Once trained, the function returns the trained KNN model. KNN is particularly effective for simple classification tasks and is sensitive to the local structure of the data, making it suitable for scenarios where the decision boundaries are nonlinear or complex. However, it's essential to choose an appropriate value for k to balance model complexity and accuracy.

![image](https://github.com/Ayoubelfallah/system_grading_islamic_questions/assets/93795377/d93da432-a6c5-411b-b2f4-1794f10311ed)

Decision Tree:

The Decision Tree method is a machine learning model belonging to the supervised learning algorithm family. It is employed for solving classification and regression problems by recursively dividing the feature space into homogeneous regions. The fundamental idea behind the Decision Tree is to make decisions by progressively splitting the dataset into smaller subsets based on the most informative features at each step. Each node in the tree represents a question about a specific feature, and the branches describe the different possible answers to that question. The algorithm learns these questions and answers from the training data, aiming to maximize the purity of the resulting subsets.

![image](https://github.com/Ayoubelfallah/system_grading_islamic_questions/assets/93795377/bce06791-f4af-4a3c-9b41-8bffe41f2fc6)

Artificial neural network:

This ANN is configured with a three-layer architecture, consisting of hidden layers with 10, 8, and 6 neurons, respectively. The max_iter parameter is set to 500, determining the maximum number of iterations for the solver to converge during training. The function then fits the ANN model to the provided training data (X_train for features and Y_train for target labels) and returns the trained model.
Artificial Neural Networks, and specifically Multi-Layer Perceptrons (MLPs), are powerful models capable of learning complex patterns in data. In this context, the hidden layers contribute to the network's ability to capture intricate relationships within the input features. The choice of the number of neurons in each layer and the maximum number of iterations is a crucial aspect of tuning the model's performance. Once trained, the Artificial Neural Network can be used for making predictions on new data points.

![image](https://github.com/Ayoubelfallah/system_grading_islamic_questions/assets/93795377/6f9a474c-dcb5-4a89-8835-f28b1e45c36e)

RNN :

This RNN architecture is designed for sequence-based tasks, such as time series prediction or natural language processing. The class constructor (_init_) initializes the model with essential parameters, including input_size (dimensionality of input features), output_size (dimensionality of the output), hidden_dim (number of hidden units in the RNN layer), and n_layers (number of RNN layers). The RNN layer is created using nn.RNN, specifying the input size, hidden dimensions, number of layers, and setting batch_first=True to expect input data with the batch size as the first dimension. The model further includes a fully connected layer (nn.Linear) to map the RNN outputs to the desired output size. Overall, this RNN architecture serves as a foundation for learning temporal dependencies in sequential data, with the flexibility to be customized for various applications through additional layers and activation functions.

![image](https://github.com/Ayoubelfallah/system_grading_islamic_questions/assets/93795377/dbe88486-b66b-43c3-8933-20a7455eda5d)

![image](https://github.com/Ayoubelfallah/system_grading_islamic_questions/assets/93795377/214f4cc0-2b2e-49e5-8e7e-b31cd0e8d8e6)

We got an accuracy around : 

![image](https://github.com/Ayoubelfallah/system_grading_islamic_questions/assets/93795377/a3e186fa-2752-4ffa-909b-16acbe0a9a5a)

As the results shows if the following plot:

![image](https://github.com/Ayoubelfallah/system_grading_islamic_questions/assets/93795377/561c230e-9280-4728-a88a-ca4487fef54c)

LSTM :

The input layer is defined with the shape (max_sequence_length,), representing the maximum length of the input sequences. An Embedding layer is employed to transform input sequences into dense vectors of fixed size, utilizing pre-trained word embeddings provided by the embedding_matrix. The Embedding layer is set to be non-trainable (trainable=False) to retain the knowledge embedded in the pre-trained word vectors.
The LSTM layer follows with 64 units, configured to return sequences and not states. This LSTM layer captures temporal dependencies in the input sequences. Dropout layers with a rate of 0.2 are introduced to prevent overfitting during training.
The subsequent Flatten layer is applied to flatten the LSTM output, preparing it for dense layers. Two Dense layers with 32 and 6 units, respectively, are utilized for further feature transformation and the final classification task. The activation functions used are Rectified Linear Unit (ReLU) for intermediate layers and Softmax for the output layer, which is common for multi-class classification problems.

The model is then defined using the Model class from Keras, specifying the input and output layers. This LSTM-based architecture is suitable for tasks that involve learning sequential patterns and making predictions in a categorical context.

![image](https://github.com/Ayoubelfallah/system_grading_islamic_questions/assets/93795377/7dfa7228-e41b-4872-a9cd-3e1bd4fa96a2)

Transformers :

the Transformer architecture used in natural language processing and sequence-to-sequence tasks. This class encapsulates a self-attention mechanism and a feedforward neural network, both crucial components of the Transformer model. The self-attention mechanism, implemented as layers.MultiHeadAttention, enables the model to focus on different parts of the input sequence, capturing long-range dependencies effectively. The feedforward neural network (keras.Sequential) processes the attended output, allowing the model to capture non-linear relationships within the data. Layer normalization and dropout layers are incorporated to stabilize and regularize the learning process. The call method defines the forward pass, where the input sequence undergoes self-attention, normalization, and feedforward processing. The TransformerBlock is a key element in creating powerful and scalable Transformer models for various sequence-based tasks.

![image](https://github.com/Ayoubelfallah/system_grading_islamic_questions/assets/93795377/82a86107-f2d6-498b-8b78-ff704191e0cb)

Testing the models And Saving best one:

![image](https://github.com/Ayoubelfallah/system_grading_islamic_questions/assets/93795377/65bd0fd6-2f14-4ef9-b2b9-4f6a3d72e380)

VI.	Insights of the Back-end making process :

Backend of our application

FastAPI  and GraphQL :

For the backend we choose FastAPI framework
because it’s fast to learn and to use.

![image](https://github.com/Ayoubelfallah/system_grading_islamic_questions/assets/93795377/69101660-859e-429e-947f-c4c7ac5ab7fc)

To send request from the frontend to backend we used GraphQL a query language for APIs and a runtime for fulfilling those queries with the existing data. 

We implemented it in our application with the library Strawberry. It’s a new GraphQL library for Python 3, inspired by dataclasses

![image](https://github.com/Ayoubelfallah/system_grading_islamic_questions/assets/93795377/78878790-4504-46da-a551-dbcb52e70b5b)

As shows the following code we get to lunch our server:

![image](https://github.com/Ayoubelfallah/system_grading_islamic_questions/assets/93795377/d10a4052-1957-4d02-9dcb-4f7028d49b7f)

VII.	Insights of the Front-end making process :

Frontend of our application

Angular

For the frontend we choose Angular and TailwindCSS framework.

![image](https://github.com/Ayoubelfallah/system_grading_islamic_questions/assets/93795377/67901689-2877-43ad-9b04-0e7cc6aad06d)

GraphQL Client Side

To use GraphQL in the client side we needed also to download a library called apollo-angular so we did using this command :
ng add apollo-angular
We needed to set the URL of our GraphQL Server in the src/app/graphql.module.ts file :
const uri = 'http://localhost:8000'; // <-- the URL of our GraphQL server
VIII.	Results and discussions: 

Home page :

![image](https://github.com/Ayoubelfallah/system_grading_islamic_questions/assets/93795377/b1a4bd02-46ef-42e2-bb2e-69227e74d9ef)

Question page :

By clicking on the button the test will begin :

![image](https://github.com/Ayoubelfallah/system_grading_islamic_questions/assets/93795377/a589bb06-2c35-41d0-afb9-a384bfbc279c)
![image](https://github.com/Ayoubelfallah/system_grading_islamic_questions/assets/93795377/ac4e0ec1-10d6-4591-a27e-2bde17b51eb3)

Result Page:

By clicking on the button the grades given by the models we trained will showed with the correct answers :

![image](https://github.com/Ayoubelfallah/system_grading_islamic_questions/assets/93795377/44fd19dd-f44d-4da2-8107-4d6002c2ac8e)

IX.	Conclusion :

This project aims to develop an Arabic Automated Grading System and Smart Assistance for Islamic education, utilizing scraping techniques to create a personalized dataset from diverse sources. The process includes establishing an Arabic NLP pipeline, performing EDA, and implementing word embedding. Model training involves RNN, LSTM, and Transformer, with text generation using BERT and fine-tuning of LLMs. Evaluation metrics such as ROC, accuracy, f1 score, and blue score are employed, leading to the deployment of the best-performing model through Docker and Kubernetes, accessible via an SAP web application. Key tools encompass NLTK, Word2vec, Glove, PyTorch, Flask/FastAPI, GraphQL, Angular, Docker, Kubernetes and GitHub. The project adopts MLOps practices for efficient logistics management.

X.	Références :

Cours DEEP LEARNING & APPLICATIONS. EL AACHAK Lotfi 2023/2024

https://islamqa.info/ar/answers/226422/%D9%81%D8%B1%D8%A7%D9%89%D8%B6-%D8%A7%D9%84%D9%88%D8%B6%D9%88%D8%A1-%D9%88%D8%B3%D9%86%D9%86%D9%87 

https://minhaji.net/lesson/21940/%D8%A5%D8%AC%D8%A7%D8%A8%D8%A7%D8%AA_%D8%A3%D8%AE%D8%AA%D8%A8%D8%B1_%D9%85%D8%B9%D9%84%D9%88%D9%85%D8%A7%D8%AA%D9%8A

https://github.com

https://chat.openai.com/

https://www.youtube.com/

https://www.kaggle.com/

https://www.google.com/search?client=firefox-b-d&q=stack+overflow

https://angular.io/

https://fastapi.tiangolo.com/

https://graphql.org/

https://medium.com/@dvelsner/deploying-a-simple-machine-learning-model-in-a-modern-web-application-flask-angular-docker-a657db075280 
