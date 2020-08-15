import nltk

'''
Tokenizing - Splitting sentences and words from the body of text.


Corpus - Body of text, singular. Corpora is the plural of this. Example: A collection of medical journals.
Lexicon - Words and their meanings. Example: English dictionary. Consider, however, that various fields will have different lexicons. For example: To a financial investor, the first meaning for the word "Bull" is someone who is confident about the market, as compared to the common English lexicon, where the first meaning for the word "Bull" is an animal. As such, there is a special lexicon for financial investors, doctors, children, mechanics, and so on.
Token - Each "entity" that is a part of whatever was split up based on rules. For examples, each word is a token when a sentence is "tokenized" into words. Each sentence can also be a token, if you tokenized the sentences out of a paragraph.

The idea of stemming is a sort of normalizing method. Many variations of words carry the same meaning, other than when tense is involved.

The reason why we stem is to shorten the lookup, and normalize sentences.

'''

#1>

##nltk.download()

#from nltk.tokenize import sent_tokenize, word_tokenize

#EXAMPLE_TEXT = "Hello Mr. Smith, how are you doing today? The weather is great, and Python is awesome. The sky is pinkish-blue. You shouldn't eat cardboard."

#print(sent_tokenize(EXAMPLE_TEXT))

#print(word_tokenize(EXAMPLE_TEXT))


#2>

##from nltk.corpus import stopwords
##from nltk.tokenize import word_tokenize
##
##example_sent = "This is a sample sentence, showing off the stop words filtration."
##
##stop_words = set(stopwords.words("english"))
##
##word_tokens = word_tokenize(example_sent)
##
###print(stop_words)
##
##filtered_sentence = [ w for w in word_tokens if not w in stop_words]
##
##print(filtered_sentence)


#3>

##from nltk.stem import PorterStemmer
##from nltk.tokenize import sent_tokenize, word_tokenize
##
##ps = PorterStemmer()
##
##exmple_words = ["python", "pythoner", "pythoned", "pythonly"]
##
##for w in exmple_words:
##    print(ps.stem(w))


#4>

'''

CHUNKING: 
Now that we know the parts of speech, we can do what is called chunking,
and group words into hopefully meaningful chunks.
One of the main goals of chunking is to group into what are known as "noun phrases." These are phrases of one or more words that contain a noun,
maybe some descriptive words, maybe a verb, and maybe something like an adverb.
The idea is to group nouns with the words that are in relation to them.
'''

from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer    

#One is a State of the Union address from 2005, and the other is from 2006 from past President George W. Bush.

##train_text = state_union.raw("2005-GWBush.txt")
##sample_text = state_union.raw("2006-GWBush.txt")
##
##custom_tokenizer = PunktSentenceTokenizer(train_text)
##
##tokenized = custom_tokenizer.tokenize(sample_text)
##
##def process_content():
##
##    try:
##        for i in tokenized:
##            words = nltk.word_tokenize(i)
##            tagged = nltk.pos_tag(words)
##            print(tagged)
##
##    except Exception as e:
##        print(str(e))
##
##process_content()


'''
POS tag list:

CC	coordinating conjunction
CD	cardinal digit
DT	determiner
EX	existential there (like: "there is" ... think of it like "there exists")
FW	foreign word
IN	preposition/subordinating conjunction
JJ	adjective	'big'
JJR	adjective, comparative	'bigger'
JJS	adjective, superlative	'biggest'
LS	list marker	1)
MD	modal	could, will
NN	noun, singular 'desk'
NNS	noun plural	'desks'
NNP	proper noun, singular	'Harrison'
NNPS	proper noun, plural	'Americans'
PDT	predeterminer	'all the kids'
POS	possessive ending	parent\'s
PRP	personal pronoun	I, he, she
PRP$	possessive pronoun	my, his, hers
RB	adverb	very, silently,
RBR	adverb, comparative	better
RBS	adverb, superlative	best
RP	particle	give up
TO	to	go 'to' the store.
UH	interjection	errrrrrrrm
VB	verb, base form	take
VBD	verb, past tense	took
VBG	verb, gerund/present participle	taking
VBN	verb, past participle	taken
VBP	verb, sing. present, non-3d	take
VBZ	verb, 3rd person sing. present	takes
WDT	wh-determiner	which
WP	wh-pronoun	who, what
WP$	possessive wh-pronoun	whose
WRB	wh-abverb	where, when

'''

'''
You may find that, after a lot of chunking, you have some words in your chunk you still do not want,
but you have no idea how to get rid of them by chunking. You may find that chinking is your solution.
Chinking is a lot like chunking, it is basically a way for you to remove a chunk from a chunk. The chunk that you remove from your chunk is your chink.

The code is very similar, you just denote the chink, after the chunk, with }{ instead of the chunk's {}.

##sentence = [("The", "DT"), ("small", "JJ"), ("red", "JJ"),("flower", "NN"), 
##("flew", "VBD"), ("through", "IN"),  ("the", "DT"), ("window", "NN")]
##
##grammar = r"""Chunk: {<NP.?>*}"""
##
##cp = nltk.RegexpParser(grammar)
##result = cp.parse(sentence) 
##print(result)
##result.draw()


'''
Here is a quick cheat sheet for various rules in regular expressions:

Identifiers:

    \d = any number
    \D = anything but a number
    \s = space
    \S = anything but a space
    \w = any letter
    \W = anything but a letter
    . = any character, except for a new line
    \b = space around whole words
    \. = period. must use backslash, because . normally means any character.

Modifiers:

    {1,3} = for digits, u expect 1-3 counts of digits, or "places"
    + = match 1 or more
    ? = match 0 or 1 repetitions.
    * = match 0 or MORE repetitions
    $ = matches at the end of string
    ^ = matches start of a string
    | = matches either/or. Example x|y = will match either x or y
    [] = range, or "variance"
    {x} = expect to see this amount of the preceding code.
    {x,y} = expect to see this x-y amounts of the precedng code

White Space Charts:

    \n = new line
    \s = space
    \t = tab
    \e = escape
    \f = form feed
    \r = carriage return

Characters to REMEMBER TO ESCAPE IF USED!

    . + * ? [ ] $ ^ ( ) { } | \

Brackets:

    [] = quant[ia]tative = will find either quantitative, or quantatative.
    [a-z] = return any lowercase letter a-z
    [1-5a-qA-Z] = return all numbers 1-5, lowercase letters a-q and uppercase A-Z
'''

import re

Sentense = ''' Kushal is 12 years old, Varun is 19 years old. '''

ages = re.findall(r'\d{1,50}' , Sentense)

words_Name = re.findall(r'[A-Z] [a-z]*' , Sentense)

print(ages)
print(words_Name)

ageDict = {}   #dictionary

x = 0

for eachName in words_Name:
    ageDict[eachName] = ages[x]
    x+=1

print(ageDict)
    























