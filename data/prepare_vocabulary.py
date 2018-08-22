import argparse
from sklearn.feature_extraction.text import CountVectorizer


def main(corpus_file, vocab_file):
    vec = CountVectorizer(input='file')
    bag_of_words = vec.fit_transform([corpus_file])
    sum_words = bag_of_words.sum(axis=0)

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    vocabulary = sorted(words_freq, key=lambda x: x[1], reverse=True)
    vocab_file.write("<S>\n")
    vocab_file.write("</S>\n")
    vocab_file.write("<UNK>\n")
    for word, count in vocabulary:
        vocab_file.write("%s\n" % word.encode('utf8'))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_file', help='Corpus file for vocabulary building')
    parser.add_argument('--output_file', help='File for vocabulary output')
    args = parser.parse_args()
    corpusFilePath = args.corpus_file
    outputFilePath = args.output_file
    corpus_file = open(corpusFilePath)
    vocab_file = open(outputFilePath, 'w')
    main(corpus_file, vocab_file)
