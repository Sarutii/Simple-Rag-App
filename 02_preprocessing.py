import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

for _pkg, _path in [("stopwords", "corpora/stopwords"), ("wordnet", "corpora/wordnet"),
                     ("punkt", "tokenizers/punkt"), ("punkt_tab", "tokenizers/punkt_tab")]:
    try:
        nltk.data.find(_path)
    except (LookupError, OSError):
        try:
            nltk.download(_pkg, quiet=True)
        except Exception:
            pass  

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    stop_words = {"the", "is", "and", "a", "an", "of", "to", "in", "for", "with", "on"}


def safe_word_tokenize(text):
    try:
        return word_tokenize(text)
    except (LookupError, OSError):
        return re.findall(r"\b\w+\b", text)


def safe_lemmatize(token, pos="v"):
    try:
        return lemmatizer.lemmatize(token, pos=pos)
    except (LookupError, OSError):
        return token


def remove_urls(text):
    return re.sub(r"http\S+|www\.\S+", " ", text)


def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_numbers(text):
    return re.sub(r"\d+", " ", text)


def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def preprocess_text(
    text,
    lowercase=True,
    remove_url=True,
    remove_punct=True,
    remove_num=False,
    normalize_space=True,
    remove_stop_words=False,
    preserve_negation=True,
    use_stemming=False,
    use_lemmatization=False,
):
    if use_stemming and use_lemmatization:
        raise ValueError("Use either stemming or lemmatization, not both at the same time.")

    if lowercase:
        text = text.lower()

    if remove_url:
        text = remove_urls(text)

    if remove_punct:
        text = remove_punctuation(text)

    if remove_num:
        text = remove_numbers(text)

    if normalize_space:
        text = normalize_whitespace(text)

    if not text:
        return ""

    tokens = safe_word_tokenize(text)

    if remove_stop_words:
        if preserve_negation:
            protected_negation_words = {"no", "not", "nor", "never"}
            tokens = [token for token in tokens if (token not in stop_words) or (token in protected_negation_words)]
        else:
            tokens = [token for token in tokens if token not in stop_words]

    if use_stemming:
        tokens = [stemmer.stem(token) for token in tokens]

    if use_lemmatization:
        tokens = [safe_lemmatize(token, pos="v") for token in tokens]

    return " ".join(tokens)


def build_search_text(doc_title, department, doc_type, chunk_text):
    return f"{doc_title} {department} {doc_type} {chunk_text}"


def tokenize(text):
    cleaned = preprocess_text(
        text,
        remove_stop_words=True,
        preserve_negation=True,
        use_lemmatization=True,
    )
    return cleaned.split()


if __name__ == "__main__":
    sample = "Students may not request an official Transcript - fee: 15 USD! See https://example.edu"
    print("default preprocess_text:", repr(preprocess_text(sample)))
    print("retrieval tokenize:     ", tokenize(sample))
