# MLThe MNIST is a widely used Data set in Machine Learning community. When I start learning ML, I try all kinds of ML models on MNIST Data set and also analyze other people’s result such as Karpathy’s Convnetjs. I notice that human makes errors at some ‘bad’ handwriting. The CNN can predict these correctly, but make errors at some ‘good’ handwriting. I prove that the ‘bad’ handwriting in training set and the CNN learn it to predict some ‘bad’ handwriting correctly with the cost of predicting some ‘good’ handwriting wrong. At same time I find some mislabeled instances in training and test data set, which proves the current the state of the art on MNIST are not accurate. By analyzing how NIST SD data set was created, I find that there are some uncertainty in NIST and also in MNIST. Since there are the mislabeled and uncertain instances in MNIST, then using only one accurate on test as criterion is not appropriate. It might be better to build another MNIST, let's call it MNIST-augmentation, which includes 60000 ground true training instances and 10000 ground true test instances, and adding 100 ground false training instances, and 1000 uncertain training instances and 1000 uncertain test instances. With such data set, we can get more accurate the result of accuracy on ground true data set and also assess the model with some ground false training instances and uncertain training instances, and how the model predict the uncertain test instances.
