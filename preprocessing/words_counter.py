from mrjob.job import MRJob
import re
import csv


class SearchDAU(MRJob):
    def mapper(self, _, line):
        if len(line.split("|")) == 5:
            index, category, title, description, text = line.split("|")

            line = title + " " + description + " " + text
            for word in re.findall("\w+", line):
                key = word.lower()

                yield ((category, key), 1)
        else:
            yield (("Не получилось", "Слов меньше 5"), 1)

    def reducer(self, key, value):
        yield key, sum(value)


def count_words(input_file, output_file):
    s = SearchDAU([input_file])
    with s.make_runner() as runner:
        runner.run()

        output = s.parse_output(runner.cat_output())

        with open(output_file, "w", newline="") as csvfile:
            spamwriter = csv.writer(csvfile)

            for line in sorted(output):
                spamwriter.writerow([line[0][0], line[0][1], line[1]])
