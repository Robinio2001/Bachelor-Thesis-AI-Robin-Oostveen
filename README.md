# Bachelor-Thesis-AI-Robin-Oostveen
Contains all code files and models created for my Bachelor's Thesis in AI on sentiment classification in the financial and medical domains.

## About
This repository was created to provide transparency on the research done on sentiment classification in the financial and medical domains.

The aim of this research was to compare the lexicon-based, traditional machine learning and deep learning approaches on the task of sentiment classification on the Financial Phrasebank and Drugs.com reviews datasets.
The lexicon-based approach was represented by the VADER lexicon, by Hutto & Gilbert (2014). 
The traditional machine learning approach was represented by a Support Vector Machine.
Finally, the deep learning approach was implemented via a CNN using BERT embeddings.

Based on the found results, it was concluded that the SVM is able to challenge the deep learning model based on performance, for only a fraction of the costs. 
However, it was also found that this is not necessarily the case for all domains, as the deep learning model was able to outperform the SVM significantly in the medical domain.
Hence, the most cost-effective model depends on the domain it is used in, and on the priorities of the user.

It is suggested that in future projects, it might be better to first start with an SVM to save time and resources, and only scale up to deep learning models if the performance is not yet satisfactory.
The results of this research might spark renewed interest in the development and modernization of the traditional machine learning techniques. In turn, the performance of these models might increase further, providing cheaper alternatives to expensive deep learning models.

## Files
financial_BERT_CNN_model: contains the files necessary to import the trained model using Tensorflow.
medical_BERT_CNN_model: contains the trained BERT+CNN model, ready to be imported using Tensorflow.

Financial Lexicon Based Approach.ipynb: contains the code for application of the VADER lexicon on the Financial Phrasebank.
Financial SVM.ipynb: contains the code for application of an SVM on the Financial Phrasebank.
Financial DL Model Tensorflow.ipynb: contains the code for the application of the BERT+CNN model on the Financial Phrasebank.

Medical Lexicon Based Approach.ipynb: contains the code for application of the VADER lexicon on the Drugs.com reviews dataset.
Medical SVM.ipynb: contains the code for application of an SVM on the Drugs.com reviews dataset.
Medical DL Tensorflow.ipynb: contains the code for the application of the BERT+CNN model on the Drugs.com reviews dataset.

Preprocessing Pipeline and Data Exploration Financial.ipynb: contains the code for preprocessing the financial data and performing data exploration on it.
Preprocessing Pipeline and Data Exploration Medical.ipynb: contains the code for preprocessing the medical data and performing data exploration on it.

Error indices: folder containing the indices of specific errors from the cells of the confusion matrix.

## Author
Robin Oostveen
6732348, Utrecht University
Supervisor: Yingjin Song
