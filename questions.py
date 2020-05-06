import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 2


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    p = os.path.dirname(os.path.abspath(directory))
    ret=dict()
    dir=os.path.join(p,directory)
    for filename in os.listdir(dir):
        fname=os.path.join(dir,filename)
        if filename[-4:] == ".txt":
            with open(fname,encoding="utf8") as f:
                ret[filename] = f.read()
    #print(ret)
    return ret


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """

    ret=[]
    for word in nltk.word_tokenize(document):
        if not word in nltk.corpus.stopwords.words("english"):
            for c in word:
                if c in string.punctuation:
                    word=word.replace(c,'')
            ret.append(word.lower())
    return ret


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    words = set()
    for filename in documents.keys():
        words.update(documents[filename])

    idfs = dict()
    for word in words:
        f = sum(word in documents[filename] for filename in documents.keys())
        idf = math.log(len(documents) / f)
        idfs[word] = idf
    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tfidfs = dict()
    for filename in files.keys():
        tfidfs[filename] = []
        for word in set(files[filename]):
            tf = files[filename].count(word)
            tfidfs[filename].append((word, tf * idfs[word]))

    ret=[]
    # Sort and get top n TF-IDFs for each file
    for filename in files.keys():
        tfidfs[filename].sort(key=lambda tfidf: tfidf[1], reverse=True)
        sum=0
        for word in query:
            if word in list(zip(*tfidfs[filename]))[0]:
                for s in tfidfs[filename]:
                    if s[0] == word:
                        sum+=s[1]
                        break
        ret.append((filename,sum))


    ret.sort(key=lambda x:x[1], reverse=True)

    r=[]
    for i in range(0,n):
        r.append(ret[i][0])

    return r


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    ret=[]
    for sentence in sentences.keys():
        lst=list(set(sentences[sentence]) & set(query))
        sum=0
        for word in lst:
            sum+=idfs[word]
        ret.append((sentence,sum, float(len(lst)/len(sentence))))

    ret=sorted(ret,key=lambda x:(x[1],x[2]), reverse=True)
    r=[]
    for i in range(0,n):
        r.append(ret[i][0])

    return r


if __name__ == "__main__":
    main()
