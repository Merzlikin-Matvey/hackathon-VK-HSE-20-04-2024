import pandas as pd
import words_counter


class ArticlesParser:
    words_count: pd.DataFrame

    def __init__(self, articles: pd.DataFrame, output_file: str = "words_coeffs.csv"):
        """
        Parameters:
            articles: (Union[pd.DataFrame, str]) - DataFrame or filename of csvfile with articles with columns: 'Class', 'Title', 'Description', 'Text'
            output_file: (str) - filename of csvfile to save words words coefficients with columns: 'word', *classes, 'total'
        """
        if isinstance(articles, str):
            articles = pd.read_csv(articles)
        self.articles = articles
        self.count_words()
        self.count_coeffs()
        self.coeffs.to_csv(output_file)

    def count_words(self):
        input_filename, output_filename = (
            "__words_counting_input.csv",
            "__words_counting_output.csv",
        )
        self.articles.to_csv(input_filename, sep="|", header=False)
        words_counter.count_words(input_filename, output_filename)
        self.words_count = pd.read_csv(output_filename)

        self.words_count.columns = ["category", "word", "count"]
        self.words_count = self.words_count.pivot_table(
            values="count", index="word", columns="category", aggfunc="sum"
        )
        self.words_count = self.words_count.fillna(0)
        self.words_count = self.words_count[~self.words_count.index.str.isnumeric()]
        self.words_count = self.words_count.drop(columns=["Не получилось"])
        self.words_count["total"] = self.words_count.sum(axis=1)

    def count_coeffs(self):
        self.coeffs = self.words_count[self.words_count["total"] > 30]
        class_columns = list(
            filter(lambda elem: elem not in ["total", "word"], self.coeffs.columns)
        )
        self.coeffs[class_columns] += 1
        self.coeffs[class_columns] = self.coeffs[class_columns].apply(
            lambda x: x / x.sum(), axis=0
        )
        self.coeffs[class_columns] = self.coeffs[class_columns].apply(
            lambda x: x / x.sum(), axis=1
        )
        self.coeffs.to_csv("words_indexes.csv")


ArticlesParser("data_processed_train.csv")
