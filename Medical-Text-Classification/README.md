# Medical Text Classification

## Text Preprocess
### Stemming
Directly converting the raw document in matrix feature representation, the total possible feature space (dictionary) will be too huge, and most of them are the inflectional form of the same word. One way to reduce is doing stemming and lemmatization. In my implementation, I used NLTK (Natural Language ToolKit) to process the lemmatization steps.

### Word representation

#### TF-IDF  and BOW 
TF-IDF (Term frequency-inverse document frequency) and BOW (bag-of-words) are common techniques using in natural language process. Since most tokens in our documents are the medical terms, directly use BOW may not reflect different medical terms in the different class document. Under this assumption, it's natural to use TF-IDF as text feature representation.

#### N-gram
A sentence has composed a series of words, the relationship between word is also significant. To capture the relationship between words, I also used 4-gram during convert document to feature representation.

### Imbalance dataset
Data sampling (over-sampling and under-sampling): 
Since the dataset is imbalanced, one way to mitigate this problem is to adjust class distribution between the major and minor class. In my implementation, I tried using one of standard technique Synthetic Minority Over-sampling (SMOTE) to synthesize sample in feature space.

#### Text synthesize
SOMTE synthesize in feature space may not mapping back to the text representation. Hence, I also tried another approach to synthesize sample text. The basic idea is generating a sequence with text with length distribution is similar to the dataset. During generating each text sample random sample token from the dataset with same length text.

#### Weighted class
The other way to overcome the imbalanced, we can try using the classifier that supports class weighing in cost function to control different penalty for various class. 

### Model
Support Vector Machine (SVM) is one of the state-of-art methods is text classification and combine weighted class penalty can easily deal with imbalance data. Our dataset covert to feature space with over 30000 features. Thus, I choose stochastic gradient descent-based implementation in sklearn as my model.

## Result
The sgd (stochastic gradient descent), imb (SMOTE sampling), syn (synthesize text) model result showed that using oversampling the recall of minor class increase thus the overall f1 score improved.

Leaderboard Rank: 1, 0.7966

sgd full macro f1 (training): 0.7713909716394991
             precision    recall  f1-score   support

          1       0.80      0.90      0.85      3163
          2       0.74      0.68      0.71      1494
          3       0.78      0.73      0.75      1925
          4       0.79      0.86      0.82      3051
          5       0.76      0.70      0.73      4805

avg / total       0.78      0.78      0.78     14438

imb full macro f1 (training): 0.850238598015445
             precision    recall  f1-score   support

          1       0.87      0.91      0.89      4805
          2       0.86      0.96      0.91      4805
          3       0.87      0.93      0.90      4805
          4       0.84      0.92      0.88      4805
          5       0.84      0.57      0.68      4805

avg / total       0.86      0.86      0.85     24025

syn full macro f1 (training): 0.8524813604124072
             precision    recall  f1-score   support

          1       0.86      0.92      0.89      4805
          2       0.88      0.94      0.91      4805
          3       0.88      0.92      0.90      4805
          4       0.85      0.91      0.88      4805
          5       0.81      0.59      0.69      4805

avg / total       0.85      0.86      0.85     24025


## Reference

Text classification with NLTK and Scikit-Learn
https://bbengfort.github.io/tutorials/2016/05/19/text-classification-nltk-sckit-learn.html

imbalanced-learn
http://contrib.scikit-learn.org/imbalanced-learn/stable/index.html

SMOTE on text classification
https://datascience.stackexchange.com/questions/27671/how-do-you-apply-smote-on-text-classification
