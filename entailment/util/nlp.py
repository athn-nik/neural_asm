import numpy
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from tqdm import tqdm
from unidecode import unidecode

LATIN_1_CHARS = (
    (b'\xe2\x80\x99', b"'"),
    (b'\xc3\xa9', b'e'),
    (b'\xe2\x80\x90', b'-'),
    (b'\xe2\x80\x91', b'-'),
    (b'\xe2\x80\x92', b'-'),
    (b'\xe2\x80\x93', b'-'),
    (b'\xe2\x80\x94', b'-'),
    (b'\xe2\x80\x94', b'-'),
    (b'\xe2\x80\x98', b"'"),
    (b'\xe2\x80\x9b', b"'"),
    (b'\xe2\x80\x9c', b'"'),
    (b'\xe2\x80\x9c', b'"'),
    (b'\xe2\x80\x9d', b'"'),
    (b'\xe2\x80\x9e', b'"'),
    (b'\xe2\x80\x9f', b'"'),
    (b'\xe2\x80\xa6', b'...'),
    (b'\xe2\x80\xb2', b"'"),
    (b'\xe2\x80\xb3', b"'"),
    (b'\xe2\x80\xb4', b"'"),
    (b'\xe2\x80\xb5', b"'"),
    (b'\xe2\x80\xb6', b"'"),
    (b'\xe2\x80\xb7', b"'"),
    (b'\xe2\x81\xba', b"+"),
    (b'\xe2\x81\xbb', b"-"),
    (b'\xe2\x81\xbc', b"="),
    (b'\xe2\x81\xbd', b"("),
    (b'\xe2\x81\xbe', b")")
)


def clean_unicode_text(text):
    # try:
    #     text = text.encode()
    #     for _hex, _char in LATIN_1_CHARS:
    #         text = text.replace(_hex, _char)
    #     return text.decode('unicode-escape')
    # except:
    #     print(text)
    # return text.encode('ascii', 'ignore').decode('unicode-escape')
    return unidecode(text).encode().decode('unicode-escape')


def tokenize(text, lowercase=True):
    if lowercase:
        text = text.lower()
    return text.split()


def twitter_preprocess():
    preprocessor = TextPreProcessor(
        normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
                   'time',
                   'date', 'number'],
        annotate={"hashtag", "elongated", "allcaps", "repeated", 'emphasis',
                  'censored'},
        all_caps_tag="wrap",
        fix_text=True,
        segmenter="twitter_2018",
        corrector="twitter_2018",
        unpack_hashtags=True,
        unpack_contractions=True,
        spell_correct_elong=False,
        tokenizer=SocialTokenizer(lowercase=True).tokenize,
        dicts=[emoticons]
    ).pre_process_doc

    def preprocess(name, dataset):
        desc = "PreProcessing dataset {}...".format(name)

        # data = []
        # with multiprocessing.Pool(processes=4) as pool:
        #     iterator = pool.imap_unordered(preprocessor, X, 1000)
        #     for i, result in enumerate(tqdm(iterator, total=len(X))):
        #         pass

        data = [preprocessor(x)
                for x in tqdm(dataset, desc=desc)]
        return data

    return preprocess


def vectorize(sequence, el2idx, max_length, unk_policy="random",
              spell_corrector=None):
    """
    Covert array of tokens, to array of ids, with a fixed length
    and zero padding at the end
    Args:
        sequence (): a list of elements
        el2idx (): dictionary of word to ids
        max_length ():
        unk_policy (): how to handle OOV words

    Returns: list of ids with zero padding at the end

    """
    words = numpy.zeros(max_length).astype(int)

    # trim tokens after max length
    sequence = sequence[:max_length]

    for i, token in enumerate(sequence):
        if token in el2idx:
            words[i] = el2idx[token]
        else:
            if unk_policy == "random":
                words[i] = el2idx["<unk>"]
            elif unk_policy == "zero":
                words[i] = 0
            elif unk_policy == "correct":
                corrected = spell_corrector(token)
                # if corrected != token:
                #     print(token, corrected)
                if corrected in el2idx:
                    words[i] = el2idx[corrected]
                else:
                    words[i] = el2idx["<unk>"]

    return words
