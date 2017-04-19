#!/usr/bin/env python3

# Copyright 2015 Conchylicultor. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Loads the dialogue corpus, builds the vocabulary
"""

import numpy as np
import nltk  # For tokenize
from tqdm import tqdm  # Progress bar
import pickle  # Saving the data
import math  # For float comparison
import os  # Checking file existance
import random
import string
from collections import OrderedDict

from chatbot.corpus.cornelldata import CornellData
from chatbot.corpus.opensubsdata import OpensubsData


# from chatbot.corpus.scotusdata import ScotusData
# from chatbot.corpus.ubuntudata import UbuntuData
# from chatbot.corpus.lightweightdata import LightweightData


class Batch:
    """Struct containing batches info
    """

    def __init__(self):
        self.encoderSeqs = []
        self.decoderSeqs = []
        self.targetSeqs = []
        self.weights = []


class TextData:
    """Dataset class
    Warning: No vocabulary limit
    """
    availableCorpus = OrderedDict([  # OrderedDict because the first element is the default choice
        ('cornell', CornellData),
        ('opensubs', OpensubsData)
        # ('scotus', ScotusData),
        # ('ubuntu', UbuntuData),
        # ('lightweight', LightweightData),
    ])

    @staticmethod
    def corpusChoices():
        """Return the dataset availables
        Return:
            list<string>: the supported corpus
        """
        return list(TextData.availableCorpus.keys())

    def __init__(self, args):
        """Load all conversations
        Args:
            args: parameters of the model
        """
        # Model parameters
        self.args = args

        # Path variables
        self.corpusDir = os.path.join(self.args.rootDir, 'data', self.args.corpus)
        self.samplesDir = os.path.join(self.args.rootDir, 'data/samples/')
        self.testSamplesDir = os.path.join(self.args.rootDir, 'data/test')
        self.samplesName = self._constructName()

        self.padToken = -1  # Padding
        self.goToken = -1  # Start of sequence
        self.eosToken = -1  # End of sequence
        self.unknownToken = -1  # Word dropped from vocabulary

        # 2d array containing each question and his answer [[input,target]]
        self.trainingSamples = []  # train set
        self.validatingSamples = []  # validate set used to calculate evaluation score on train samples
        self.developSamples = []  # development set used to choose best parameters and re-training
        self.testingSamples = [] # test set

        self.word2id = {}
        self.id2word = {}  # For a rapid conversion

        self.loadCorpus(self.samplesDir, self.testSamplesDir)

        # Plot some stats:
        print('Loaded {} : {} words, {} QA in training set'.format(self.args.corpus, len(self.word2id), len(self.trainingSamples)))

        if self.args.playDataset:
            self.playDataset()

    def _constructName(self):
        """Return the name of the dataset that the program should use with the current parameters.
        Computer from the base name, the given tag (self.args.datasetTag) and the sentence length
        """
        baseName = 'dataset-{}'.format(self.args.corpus)
        if self.args.datasetTag:
            baseName += '-' + self.args.datasetTag
        if self.args.historyInputs:
            baseName += '-h'
        return '{}-{}.pkl'.format(baseName, self.args.maxLength)

    def makeLighter(self, ratioDataset):
        """Only keep a small fraction of the dataset, given by the ratio
        """
        # if not math.isclose(ratioDataset, 1.0):
        #     self.shuffle()  # Really ?
        #     print('WARNING: Ratio feature not implemented !!!')
        pass

    def shuffle(self):
        """Shuffle the training samples
        """
        print('Shuffling the training set...')
        random.shuffle(self.trainingSamples)

    def vld_shuffle(self):
        """Shuffle the validating samples
        """
        print('Shuffling the validating set...')
        random.shuffle(self.validatingSamples)

    def _createBatch(self, samples):
        """Create a single batch from the list of sample. The batch size is automatically defined by the number of
        samples given.
        The inputs should already be inverted. The target should already have <go> and <eos>
        Warning: This function should not make direct calls to args.batchSize !!!
        Args:
            samples (list<Obj>): a list of samples, each sample being on the form [input, target]
        Return:
            Batch: a batch object en
        """

        batch = Batch()
        batchSize = len(samples)

        # Create the batch tensor
        for i in range(batchSize):
            # Unpack the sample
            sample = samples[i]
            if not self.args.test and self.args.watsonMode:  # Watson mode: invert question and answer
                sample = list(reversed(sample))
            batch.encoderSeqs.append(list(reversed(
                sample[0])))  # Reverse inputs (and not outputs), little trick as defined on the original seq2seq paper
            batch.decoderSeqs.append([self.goToken] + sample[1] + [self.eosToken])  # Add the <go> and <eos> tokens
            batch.targetSeqs.append(
                batch.decoderSeqs[-1][1:])  # Same as decoder, but shifted to the left (ignore the <go>)

            # Long sentences should have been filtered during the dataset creation
            assert len(batch.encoderSeqs[i]) <= self.args.maxLengthEnco
            assert len(batch.decoderSeqs[i]) <= self.args.maxLengthDeco

            # Add padding & define weight
            batch.encoderSeqs[i] = [self.padToken] * (self.args.maxLengthEnco - len(batch.encoderSeqs[i])) + \
                                   batch.encoderSeqs[i]  # Left padding for the input
            batch.weights.append(
                [1.0] * len(batch.targetSeqs[i]) + [0.0] * (self.args.maxLengthDeco - len(batch.targetSeqs[i])))
            batch.decoderSeqs[i] = batch.decoderSeqs[i] + [self.padToken] * (
                self.args.maxLengthDeco - len(batch.decoderSeqs[i]))
            batch.targetSeqs[i] = batch.targetSeqs[i] + [self.padToken] * (
                self.args.maxLengthDeco - len(batch.targetSeqs[i]))

        # Simple hack to reshape the batch
        # 对encoderSeqs做转置，使得encoderSeqs[i]代表多个句子的第i个单词，len(encoderSeqs)=maxLengthEnco而不是batchSize
        encoderSeqsT = []  # Corrected orientation
        for i in range(self.args.maxLengthEnco):
            encoderSeqT = []
            for j in range(batchSize):
                encoderSeqT.append(batch.encoderSeqs[j][i])
            encoderSeqsT.append(encoderSeqT)
        batch.encoderSeqs = encoderSeqsT

        decoderSeqsT = []
        targetSeqsT = []
        weightsT = []
        for i in range(self.args.maxLengthDeco):
            decoderSeqT = []
            targetSeqT = []
            weightT = []
            for j in range(batchSize):
                decoderSeqT.append(batch.decoderSeqs[j][i])
                targetSeqT.append(batch.targetSeqs[j][i])
                weightT.append(batch.weights[j][i])
            decoderSeqsT.append(decoderSeqT)
            targetSeqsT.append(targetSeqT)
            weightsT.append(weightT)
        batch.decoderSeqs = decoderSeqsT
        batch.targetSeqs = targetSeqsT
        batch.weights = weightsT

        # # Debug
        # self.printBatch(batch)  # Input inverted, padding should be correct
        # print(self.sequence2str(samples[0][0]))
        # print(self.sequence2str(samples[0][1]))  # Check we did not modified the original sample

        return batch

    def getBatches(self, validate=0):
        """Prepare the batches for the current epoch
        Return:
            list<Batch>: Get a list of the batches for the next epoch
        """
        if validate == 0:
            self.shuffle()
        else:
            if not len(self.validatingSamples) == 0:
                self.validatingSamples.clear()
            if validate == len(self.trainingSamples):
                self.validatingSamples.extend(self.trainingSamples)
            else:
                self.validatingSamples.extend(random.sample(self.trainingSamples, validate))
            self.vld_shuffle()

        batches = []

        def genNextSamples(vld):
            """ Generator over the mini-batch training samples
            """
            if vld == 0:
                for i in range(0, self.getSampleSize()[0], self.args.batchSize):
                    # i取0-SampleSize中的值，循环间取值间隔为batchSize
                    yield self.trainingSamples[i:min(i + self.args.batchSize, self.getSampleSize()[0])]
                    # 取trainingSamples集合中i->j的数据，j为i+batchSize和SampleSize中较小的值，只有在取最后一组数据时，
                    # 可能剩余的总数据数小于batchsize，那么就取到最大SampleSize即可
            else:
                for i in range(0, self.getSampleSize()[1], 1):
                    yield self.validatingSamples[i:min(i + 1, self.getSampleSize()[1])]

        # samples为一组qa对，大小由batchSize和getSampleSize返回值两个参数决定
        for samples in genNextSamples(validate):
            # 一个batch（也就是一次性训练数据量）由一个samples，即一组qa对组成
            if validate == 0:
                batch = self._createBatch(samples)
            else:
                batch = samples[0]
            # batches包含所有数据，是batch的集合，依次训练即可，将batches训练完就遍历了一遍数据
            batches.append(batch)
        return batches

    def getSampleSize(self):
        """Return the size of the dataset
        Return:
            list: Size of 4 type samples set, 0 for train, 1 for validate, 2 for develop, 3 for test
        """
        sample_size = [len(self.trainingSamples), len(self.validatingSamples), len(self.developSamples), len(self.testingSamples)]
        return sample_size

    def getVocabularySize(self):
        """Return the number of words present in the dataset
        Return:
            int: Number of word on the loader corpus
        """
        return len(self.word2id)

    def loadCorpus(self, dirName, testDirName = None):
        """Load/create the conversations data
        Args:
            dirName (str): The directory where to load/save the model
        """
        datasetExist = False
        if os.path.exists(os.path.join(dirName, self.samplesName)):
            datasetExist = True

        if not datasetExist:  # First time we load the database: creating all files
            print('Training samples not found. Creating dataset...')

            optionnal = ''
            if self.args.corpus == 'lightweight' and not self.args.datasetTag:
                raise ValueError('Use the --datasetTag to define the lightweight file to use.')
            else:
                optionnal = '/' + self.args.datasetTag  # HACK: Forward the filename

            # Corpus creation
            corpusData = TextData.availableCorpus[self.args.corpus](self.corpusDir + optionnal)

            self.createCorpus(corpusData.getConversations())

            # Saving
            print('Saving dataset...')
            if not testDirName:
                self.saveDataset(dirName)  # Saving tf samples
            else:
                self.saveDataset(dirName, testDirName)
        else:
            print('Loading dataset from {}...'.format(dirName))
            self.loadDataset(dirName)

        assert self.padToken == 0

    def loadTestData(self, testDirName):
        """Load/create the conversations data
        Args:
            dirName (str): The directory where to load/save the model
        """
        datasetExist = False
        if os.path.exists(os.path.join(testDirName, 'test'+self.samplesName)):
            datasetExist = True

        if not datasetExist:  # First time we load the database: creating all files
            print('Testing samples not found. Use default samples and ignore valuations.')
            return False
        else:
            print('Loading dataset from {}...'.format(testDirName))
            with open(os.path.join(testDirName, 'test'+self.samplesName), 'rb') as handle:
                data = pickle.load(handle)
                self.testingSamples = data["testingSamples"]

        assert self.padToken == 0

        return True

    def saveDataset(self, dirName, testDirName = None):
        """Save samples to file
        Args:
            dirName (str): The directory where to load/save the model
        """

        with open(os.path.join(dirName, self.samplesName), 'wb') as handle:
            data = {  # Warning: If adding something here, also modifying loadDataset
                "word2id": self.word2id,
                "id2word": self.id2word,
                "trainingSamples": self.trainingSamples,
                # "validatingSamples": self.validatingSamples
            }
            pickle.dump(data, handle, -1)  # Using the highest protocol available
        if testDirName:
            with open(os.path.join(testDirName, 'test'+self.samplesName), 'wb') as t_handle:
                testData = {
                    "testingSamples": self.testingSamples
                }
                pickle.dump(testData, t_handle, -1)  # Using the highest protocol available

    def loadDataset(self, dirName):
        """Load samples from file
        Args:
            dirName (str): The directory where to load the model
        """
        with open(os.path.join(dirName, self.samplesName), 'rb') as handle:
            data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
            self.word2id = data["word2id"]
            self.id2word = data["id2word"]
            self.trainingSamples = data["trainingSamples"]
            # self.validatingSamples = data['validatingSamples']

            self.padToken = self.word2id["<pad>"]
            self.goToken = self.word2id["<go>"]
            self.eosToken = self.word2id["<eos>"]
            self.unknownToken = self.word2id["<unknown>"]  # Restore special words

    def createCorpus(self, conversations):
        """Extract all data from the given vocabulary
        """
        # Add standard tokens
        self.padToken = self.getWordId("<pad>")  # Padding (Warning: first things to add > id=0 !!)
        self.goToken = self.getWordId("<go>")  # Start of sequence
        self.eosToken = self.getWordId("<eos>")  # End of sequence
        self.unknownToken = self.getWordId("<unknown>")  # Word dropped from vocabulary

        # Preprocessing data

        testNum = len(conversations)/10
        if not self.args.historyInputs:
            count = 1

            for conversation in tqdm(conversations, desc="Extract conversations"):
                if count <= testNum:
                    self.extractConversation(conversation, isTestSample=True)
                else:
                    self.extractConversation(conversation)
                count += 1
                # The dataset will be saved in the same order it has been extracted
            # self.validatingSamples.extend(random.sample(self.trainingSamples, 2000))
        else:
            counts = 1

            for conversation in tqdm(conversations, desc="Extract conversations"):
                if counts <= testNum:
                    self.extractHistoryConversation(conversation, isTestSample=True)
                else:
                    self.extractHistoryConversation(conversation)
                counts += 1
            for conversation in tqdm(conversations, desc="Extract conversations"):
                self.extractHistoryConversation(conversation)
            # The dataset will be saved in the same order it has been extracted

    def extractConversation(self, conversation, isTestSample = False):
        """Extract the sample lines from the conversations
        Args:
            conversation (Obj): a conversation object containing the lines to extract
        """

        # Iterate over all the lines of the conversation
        for i in tqdm_wrap(range(len(conversation['lines']) - 1),  # We ignore the last line (no answer for it)
                           desc='Conversation', leave=False):
            inputLine = conversation["lines"][i]
            targetLine = conversation["lines"][i + 1]

            inputWords = self.extractText(inputLine["text"])
            targetWords = self.extractText(targetLine["text"], isTarget=True)

            if inputWords and targetWords:  # Filter wrong samples (if one of the list is empty)
                if not isTestSample:
                    self.trainingSamples.append([inputWords, targetWords])
                else:
                    self.testingSamples.append([inputWords, targetWords])

    def extractHistoryConversation(self, conversation, method=1,isTestSample = False):
        """Extract the sample lines from the conversations
        Args:
            conversation (Obj): a conversation object containing the lines to extract
        """
        # TODO 1: Check the correctness
        # Iterate over all the lines of the conversation
        for i in tqdm_wrap(range(len(conversation['lines'])-2),
                           desc='Conversation', leave=False):
            inputLine1 = conversation["lines"][i]
            inputLine2 = conversation["lines"][i + 1]
            targetLine = conversation["lines"][i + 2]
            if method == 1:
                inputWords1 = self.extractText(inputLine1["text"])
                inputWords2 = self.extractText(inputLine2["text"])

                inputWords1.extend(inputWords2)
            elif method == 2:
                inputWords1 = self.extractText(inputLine1["text"]+' '+inputLine2["text"], ishistory=True)

            targetWords = self.extractText(targetLine["text"], isTarget=True)

            if inputWords1 and targetWords:  # Filter wrong samples (if one of the list is empty)
                if not isTestSample:
                    self.trainingSamples.append([inputWords1, targetWords])
                else:
                    self.testingSamples.append([inputWords1, targetWords])

    def extractText(self, line, isTarget=False, ishistory = False):
        """Extract the words from a sample lines
        Args:
            line (str): a line containing the text to extract
            isTarget (bool): Define the question on the answer
        Return:
            list<int>: the list of the word ids of the sentence
        """
        words = []

        # Extract sentences
        sentencesToken = nltk.sent_tokenize(line)

        # We add sentence by sentence until we reach the maximum length
        for i in range(len(sentencesToken)):
            # If question: we only keep the last sentences
            # If answer: we only keep the first sentences
            if not isTarget:
                i = len(sentencesToken) - 1 - i

            tokens = nltk.word_tokenize(sentencesToken[i])

            # If the total length is not too big, we still can add one more sentence
            if not ishistory:
                hold = self.args.maxLength
            else:
                hold = 2 * self.args.maxLength
            if len(words) + len(tokens) <= hold:
                tempWords = []
                for token in tokens:
                    # TODO 1: create vocabulary  in limited size
                    tempWords.append(self.getWordId(token))  # Create the vocabulary and the training sentences

                if isTarget:
                    words = words + tempWords
                else:
                    words = tempWords + words
            else:
                break  # We reach the max length already

        return words

    def getWordId(self, word, create=True):
        """Get the id of the word (and add it to the dictionary if not existing). If the word does not exist and
        create is set to False, the function will return the unknownToken value
        Args:
            word (str): word to add
            create (Bool): if True and the word does not exist already, the world will be added
        Return:
            int: the id of the word created
        """
        # Should we Keep only words with more than one occurrence ?
        # TODO 1: create vocabulary  in limited size
        word = word.lower()  # Ignore case

        # Get the id if the word already exist
        wordId = self.word2id.get(word, -1)

        # If not, we create a new entry
        if wordId == -1:
            if create:
                wordId = len(self.word2id)
                self.word2id[word] = wordId
                self.id2word[wordId] = word
            else:
                wordId = self.unknownToken

        return wordId

    def printBatch(self, batch):
        """Print a complete batch, useful for debugging
        Args:
            batch (Batch): a batch object
        """
        print('----- Print batch -----')
        for i in range(len(batch.encoderSeqs[0])):  # Batch size
            print('Encoder: {}'.format(self.batchSeq2str(batch.encoderSeqs, seqId=i)))
            print('Decoder: {}'.format(self.batchSeq2str(batch.decoderSeqs, seqId=i)))
            print('Targets: {}'.format(self.batchSeq2str(batch.targetSeqs, seqId=i)))
            print('Weights: {}'.format(
                ' '.join([str(weight) for weight in [batchWeight[i] for batchWeight in batch.weights]])))

    def sequence2str(self, sequence, clean=False, reverse=False):
        """Convert a list of integer into a human readable string
        Args:
            sequence (list<int>): the sentence to print
            clean (Bool): if set, remove the <go>, <pad> and <eos> tokens
            reverse (Bool): for the input, option to restore the standard order
        Return:
            str: the sentence
        """

        if not sequence:
            return ''

        if not clean:
            return ' '.join([self.id2word[idx] for idx in sequence])

        sentence = []
        for wordId in sequence:
            if wordId == self.eosToken:  # End of generated sentence
                break
            elif wordId != self.padToken and wordId != self.goToken:
                sentence.append(self.id2word[wordId])

        if reverse:  # Reverse means input so no <eos> (otherwise pb with previous early stop)
            sentence.reverse()

        return self.detokenize(sentence)

    def detokenize(self, tokens):
        """Slightly cleaner version of joining with spaces.
        Args:
            tokens (list<string>): the sentence to print
        Return:
            str: the sentence
        """
        return ''.join([
                           ' ' + t if not t.startswith('\'') and
                                      t not in string.punctuation
                           else t
                           for t in tokens]).strip().capitalize()

    def batchSeq2str(self, batchSeq, seqId=0, **kwargs):
        """Convert a list of integer into a human readable string.
        The difference between the previous function is that on a batch object, the values have been reorganized as
        batch instead of sentence.
        Args:
            batchSeq (list<list<int>>): the sentence(s) to print
            seqId (int): the position of the sequence inside the batch
            kwargs: the formatting options( See sequence2str() )
        Return:
            str: the sentence
        """
        sequence = []
        for i in range(len(batchSeq)):  # Sequence length
            sequence.append(batchSeq[i][seqId])
        return self.sequence2str(sequence, **kwargs)

    def sentence2enco(self, sentence, ishistory=False):
        """Encode a sequence and return a batch as an input for the model
        Return:
            Batch: a batch object containing the sentence, or none if something went wrong
        """

        if sentence == '':
            return None

        # First step: Divide the sentence in token
        tokens = nltk.word_tokenize(sentence)
        if not ishistory:
            hold = self.args.maxLength
        else:
            hold = 2*self.args.maxLength
        if len(tokens) > hold:
            return None

        # Second step: Convert the token in word ids
        wordIds = []
        for token in tokens:
            wordIds.append(self.getWordId(token, create=False))  # Create the vocabulary and the training sentences

        # Third step: creating the batch (add padding, reverse)
        batch = self._createBatch([[wordIds, []]])  # Mono batch, no target output

        return batch

    def deco2sentence(self, decoderOutputs):
        """Decode the output of the decoder and return a human friendly sentence
        decoderOutputs (list<np.array>):
        """
        sequence = []

        # Choose the words with the highest prediction score
        for out in decoderOutputs:
            sequence.append(np.argmax(out))  # Adding each predicted word ids
            # 每一个out代表一个位置(共12个，10+<go>+<eos>)的预测矩阵（矩阵大小对应词表大小），np.argmax选择矩阵中最大概率的词,并返回其下标即word ids
        return sequence  # We return the raw sentence. Let the caller do some cleaning eventually

    def playDataset(self):
        """Print a random dialogue from the dataset
        """
        print('Randomly play samples:')
        for i in range(self.args.playDataset):
            idSample = random.randint(0, len(self.trainingSamples) - 1)
            print('Q: {}'.format(self.sequence2str(self.trainingSamples[idSample][0], clean=True)))
            print('A: {}'.format(self.sequence2str(self.trainingSamples[idSample][1], clean=True)))
            print()
        pass


def tqdm_wrap(iterable, *args, **kwargs):
    """Forward an iterable eventually wrapped around a tqdm decorator
    The iterable is only wrapped if the iterable contains enough elements
    Args:
        iterable (list): An iterable object which define the __len__ method
        *args, **kwargs: the tqdm parameters
    Return:
        iter: The iterable eventually decorated
    """
    if len(iterable) > 100:
        return tqdm(iterable, *args, **kwargs)
    return iterable
