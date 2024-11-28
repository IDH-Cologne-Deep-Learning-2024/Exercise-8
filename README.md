# Exercise 8

This is the eighth exercise for the course [Deep Learning](https://lehre.idh.uni-koeln.de/lehrveranstaltungen/wintersemester-2024-2025/deep-learning/).

Clone this repository to your local setup. Create a branch with the name of your GitHub account and switch to it.

In the repository, you find a TSV file with science paper abstracts and associated domains ([Source](https://data.mendeley.com/datasets/9rw3vkcfy4/2))

Train a feed-forward neural network with Keras to predict the domain of an abstract.
Use and implement the training strategies presented in the [lecture](https://lehre.idh.uni-koeln.de/site/assets/files/5372/presentation08.pdf).
If you want to use pre-trained embeddings, there is a function in the code that reads in GLOVE embeddings and creates a word embedding matrix that can be provided to Keras.

Let the model predict on the test set and compute Accuracy, Precision, Recall and F1-score.
Try to get an F1 score as high as possible.

Commit your changes and push them to the branch with the name of your GitHub account.

Deadline for this exercise is December 5, 2024, 08:00:00 CET.
