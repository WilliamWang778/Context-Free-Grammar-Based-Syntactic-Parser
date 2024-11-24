import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

# NP: (Det) (AP) N (PP)
# VP: V (NP) (PP) (Adv)
#     V (NP) (Adv) (PP)
#     (Adv) V (NP) (PP)
NONTERMINALS = """
S -> NP VP
S -> S Conj S
NP -> N | Det N | AP N | Det AP N
NP -> N PP | Det N PP | AP N PP | Det AP N PP
AP -> Adj | Adj AP
VP -> V | V NP | V PP | V Adv
VP -> V PP Adv | V NP Adv | V NP PP | V NP PP Adv | V NP Adv PP
VP -> Adv V | Adv V NP | Adv V PP | Adv V NP PP
VP -> VP Conj VP 
PP -> P NP
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    res = []
    for word in nltk.word_tokenize(sentence):
        for c in word:
            if not c.isalpha():
                break
        else:
            res.append(word.lower())
    return res


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    def is_minimum_np(t):
        return t.label() == "NP" and \
               not any((is_minimum_np(sub) for sub in t.subtrees(lambda x: x != t)))

    res = list(tree.subtrees(lambda x: is_minimum_np(x)))
    return res


if __name__ == "__main__":
    main()
