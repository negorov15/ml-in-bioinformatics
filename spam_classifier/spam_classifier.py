import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def construct_corpus(data):
    """
    np.array[str, str] -> dict[str:int]
    constructs a corpus of unique words and their corresponding column indices
    from a 2D array of str, return a hash table
    """
    ##Your code here
    corpus = {}
    counter = 0

    for samples in data:
        content = samples[1].split()
        for word in content:
            # Count occurences
            # Save the first occurence of each word
            if word not in corpus:
                corpus[word] = counter
                counter += 1
    # most_common = Counter(occurrences).most_common(10)
    return corpus


def recode_messages(data, corpus):
    """
    np.array[str, str] * dict[str:int] -> np.array[int, int]

    returns the binary matrix encoding
    """
    # Your code here
    n_rows = len(data)
    n_columns = len(corpus)
    recode_matrix = np.zeros((n_rows, n_columns), dtype=int)

    # Iterate through dataset: O(n)
    for row_i in range(n_rows):
        content = data[row_i][1].split()
        # Iterate through sentence: O(n)
        for word in content:
            # Find an element in the set: O(1)
            if word in corpus:
                col_i = corpus[word]
                recode_matrix[row_i, col_i] = 1
    return recode_matrix


def train_test_split(X, Y, train_percentage=0.8, seed=1):
    """
    Splits the dataset into training and testing sets based on the specified percentage.
    """
    assert X.shape[0] == Y.shape[0]
    np.random.seed(seed)

    number_examples = X.shape[0]
    num_train = int(train_percentage * number_examples)

    train_rows = np.random.choice(number_examples, num_train, replace=False)
    test_rows = np.setdiff1d(np.arange(number_examples), train_rows)

    X_train = X[train_rows]
    Y_train = Y[train_rows]
    X_test = X[test_rows]
    Y_test = Y[test_rows]

    return X_train, Y_train, X_test, Y_test


def compute_priors(labels):
    """
    np.array[str, str] -> float, float

    computes the prior probabilities of each class (ham and spam) in the dataset
    """

    # Count occurrences of each class label
    unique, counts = np.unique(labels, return_counts=True)

    # Create a dictionary of class counts
    class_counts = dict(zip(unique, counts))

    n_observations = len(labels)
    prior_1 = class_counts["ham"] / n_observations
    prior_2 = class_counts["spam"] / n_observations

    return prior_1, prior_2


def estimate_proportions(data_matrix, Y):
    """
    estimate the matrix theta
    """
    spam_matrix = data_matrix[np.where(Y == "spam")[0]]
    ham_matrix = data_matrix[np.where(Y == "ham")[0]]

    n_spam = spam_matrix.shape[0]
    n_ham = ham_matrix.shape[0]

    spam_theta = np.sum(spam_matrix, axis=0)
    spam_theta = (spam_theta + 1) / (n_spam + 2)

    ham_theta = np.sum(ham_matrix, axis=0)
    ham_theta = (ham_theta + 1) / (n_ham + 2)

    theta = np.vstack([ham_theta, spam_theta]).T

    return theta


def classify_map(X, model):
    """
    Predicts the probability of being 'spam' for each message in X using the MAP classifier.
     - X: np.array of shape (n_samples, n_features) with binary features
     - model: tuple containing (prior_ham, prior_spam) and theta matrix
    """
    (prior_ham, prior_spam), theta = model
    # Use np.log and np.sum to avoid numerical underflow
    log_prob_spam = np.log(prior_spam) + np.sum(
        X * np.log(theta[:, 1]) + (1 - X) * np.log(1 - theta[:, 1]), axis=1
    )

    log_prob_ham = np.log(prior_ham) + np.sum(
        X * np.log(theta[:, 0]) + (1 - X) * np.log(1 - theta[:, 0]), axis=1
    )

    return [log_prob_ham, log_prob_spam]


def predict_binary(X, model):
    """
    Direct binary classification
    """

    prior, theta = model

    log_likelihood = np.sum(
        X * np.log(theta[:, 0] / theta[:, 1])
        + (1 - X) * np.log((1 - theta[:, 0]) / (1 - theta[:, 1])),
        axis=1,
    )
    log_probs = log_likelihood + np.log(prior[0] / prior[1])
    log_probs = np.where(log_probs > 0, 0, 1)

    return log_probs


def predict_posterior(log_probs):
    """
    Converts log probabilities to posterior probabilities using the softmax function.
    """
    prob = np.exp(log_probs[1]) / (np.exp(log_probs[0]) + np.exp(log_probs[1]))
    return prob


def calc_roc_curve(y_true, y_pred):
    """
    Calculates the true positive rate (TPR) and false positive rate (FPR) for different thresholds.
    """
    # sort arrays
    pred_idxs = np.argsort(y_pred)[::-1]
    y_pred = y_pred[pred_idxs]
    y_true = y_true[pred_idxs]

    distinct_idxs = np.where(np.diff(y_pred))[0]
    threshold_idxs = np.r_[distinct_idxs, y_true.size - 1]

    tpc = np.cumsum(y_true)[threshold_idxs]
    fpc = 1 + threshold_idxs - tpc

    tpr = np.r_[0, tpc] / tpc[-1]
    fpr = np.r_[0, fpc] / fpc[-1]

    return tpr, fpr


if __name__ == "__main__":
    # load dataset
    sms_data = np.loadtxt(
        "Spam Classifier/SMSSpamCollection_cleaned.csv",
        delimiter="\t",
        skiprows=1,
        dtype=str,
    )
    # Construct the corpus and recode the messages
    D = construct_corpus(sms_data)
    sms_matrix = recode_messages(sms_data, D)
    sms_labels = sms_data[:, 0]
    sms_labels = np.array([label.strip() for label in sms_labels])

    # plot frequency of words in ham and spam messages
    theta = estimate_proportions(sms_matrix, sms_labels)
    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    x = np.arange(theta.shape[0])
    for i, ax in enumerate(axs):
        word_freq = theta[:, i]

        ax.bar(x, word_freq, color="k")
        ax.set_title(f"Word frequencies for class {i}")

        ax.set_ylim(0, np.max(theta) * 1.1)
    plt.tight_layout()
    plt.savefig("word_frequencies.png")

    # Train and test the model
    X, Y = sms_matrix, sms_labels
    train_X, train_Y, test_X, test_Y = train_test_split(X, Y)
    # Train the model
    prior_ham, prior_spam = compute_priors(train_Y)
    theta = estimate_proportions(train_X, train_Y)
    model = ((prior_ham, prior_spam), theta)
    # Test the model
    res = classify_map(test_X, model)
    res_binary = predict_binary(test_X, model)
    Y_test_b = np.where(test_Y == "ham", 0, 1)
    # Calculate TPR and FPR
    tpr = np.sum(Y_test_b[res_binary == 1]) / np.sum(Y_test_b)
    print(f"TPR (Sensitivity): {tpr:.3f}")
    print(f"FPR: {1-tpr:.3f}")
    # Plot ROC curve
    log_probs = classify_map(test_X, model)
    probs = predict_posterior(log_probs)
    tpr, fpr = calc_roc_curve(Y_test_b, probs)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"Naive Bayes")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("roc_curve.png")
