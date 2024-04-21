from src.text_analytics import get_most_important_words

if __name__ == "__main__":
    X, y = prepare_data_for_nn(1)
    print(X)
    print(y)
    print(X.shape)
    print(y.shape)
    print("Done!")