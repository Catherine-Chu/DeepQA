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
Main script. See README.md for more information

Use python 3
"""

import argparse  # Command line parsing
# python 2.7  import ConfigParser as configparser
import configparser  # python 3 Saving the models parameters
import datetime  # Chronometer
import os  # Files management
import numpy as np
import math
import tensorflow as tf
import nltk

from tqdm import tqdm  # Progress bar
from nltk.translate import bleu_score

from chatbot.textdata import TextData
from chatbot.model import Model
from nltk.translate import bleu_score

from chatbot.textdata import Batch
from nltk.translate import gleu_score



class Chatbot:
    """
    Main class which launch the training or testing mode
    """

    class TestMode:
        """ Simple structure representing the different testing modes
        """
        ALL = 'all'
        INTERACTIVE = 'interactive'  # The user can write his own questions
        DAEMON = 'daemon'  # The chatbot runs on background and can regularly be called to predict something

    class TrainMode:
        """ Simple structure representing the different RL training methods
        """
        SEQ2SEQ = 'seq2seq'
        MUTUALINFO = 'mutualinfo'
        REINFORCEMENT = 'reinforcement'

    class ModelTag:
        SEQ2SEQ_TAG = 'seq2seq'
        BACKWARD_TAG = 'backward'
        POLICY_RL_TAG = 'policyRL'

    def __init__(self):
        """
        """
        # Model/dataset parameters
        self.args = None

        # Task specific object
        self.textData = None  # Dataset
        self.model = None  # Sequence to sequence model
        self.vldmodel = None
        self.backward = None
        self.forward = None

        # Tensorflow utilities for convenience saving/logging
        self.writer = None
        self.saver = None
        self.modelDir = ''  # Where the model is saved
        self.globStep = 0  # Represent the number of iteration for the current model

        # TensorFlow main session (we keep track for the daemon)
        self.sess = None
        self.vldsess = None
        self.for2sess = None
        self.backsess = None

        # Filename and directories constants
        self.MODEL_DIR_BASE = 'save/model'
        self.MODEL_NAME_BASE = 'model'
        self.MODEL_EXT = '.ckpt'
        self.CONFIG_FILENAME = 'params.ini'
        self.PERFORMANCE_FILENAME = 'performance.log'
        self.CONFIG_VERSION = '0.4'
        self.TEST_IN_NAME = 'data/test/samples.txt'
        self.TEST_OUT_SUFFIX = '_predictions.txt'
        self.SENTENCES_PREFIX = ['Q: ', 'A: ']

    @staticmethod
    def parseArgs(args):
        """
        Parse the arguments from the given command line
        Args:
            args (list<str>): List of arguments to parse. If None, the default sys.argv will be parsed
        """

        parser = argparse.ArgumentParser()

        # Global options
        globalArgs = parser.add_argument_group('Global options')
        globalArgs.add_argument('--test',
                                nargs='?',
                                choices=[Chatbot.TestMode.ALL, Chatbot.TestMode.INTERACTIVE, Chatbot.TestMode.DAEMON],
                                const=Chatbot.TestMode.ALL, default=None,
                                help='if present, launch the program try to answer all sentences from data/test/ with'
                                     ' the defined model(s), in interactive mode, the user can wrote his own sentences,'
                                     ' use daemon mode to integrate the chatbot in another program')
        globalArgs.add_argument('--train',
                                nargs='?',
                                choices=[Chatbot.TrainMode.SEQ2SEQ, Chatbot.TrainMode.MUTUALINFO,
                                         Chatbot.TrainMode.REINFORCEMENT],
                                const=Chatbot.TrainMode.SEQ2SEQ, default=None,
                                help='if present, launch the program try to train the model based on pre-trained model(s)(if exist);'
                                     'in seq2seq mode, train the model with default method;'
                                     'in multualinfo mode, train the model with multual information score object function;'
                                     'in reinforcement mode, train the model with reinforcement method between 2 agent')
        globalArgs.add_argument('--createDataset', action='store_true',
                                help='if present, the program will only generate the dataset from the corpus (no training/testing)')
        globalArgs.add_argument('--playDataset', type=int, nargs='?', const=10, default=None,
                                help='if set, the program  will randomly play some samples(can be use conjointly with createDataset if this is the only action you want to perform)')
        globalArgs.add_argument('--reset', action='store_true',
                                help='use this if you want to ignore the previous model present on the model directory (Warning: the model will be destroyed with all the folder content)')
        globalArgs.add_argument('--verbose', action='store_true',
                                help='When testing, will plot the outputs at the same time they are computed')
        globalArgs.add_argument('--keepAll', type=int, default=10,
                                help='If this option is set, limited size saved model will be keep (Warning: make sure you have enough free disk space or increase saveEvery)')
        globalArgs.add_argument('--modelTag', nargs='?',
                                choices=[Chatbot.ModelTag.SEQ2SEQ_TAG, Chatbot.ModelTag.BACKWARD_TAG,
                                         Chatbot.ModelTag.POLICY_RL_TAG],
                                const=Chatbot.ModelTag.SEQ2SEQ_TAG, default=None,
                                help='tag to differentiate which model to store/load')
        globalArgs.add_argument('--rootDir', type=str, default=None,
                                help='folder where to look for the models and data')
        globalArgs.add_argument('--watsonMode', action='store_true',
                                help='Inverse the questions and answer when training (the network try to guess the question)')
        globalArgs.add_argument('--device', type=str, default=None,
                                help='\'gpu<i>\' or \'cpu\' (Warning: make sure you have enough free RAM), allow to choose on which hardware run the model')
        globalArgs.add_argument('--seed', type=int, default=None, help='random seed for replication')

        # Dataset options
        datasetArgs = parser.add_argument_group('Dataset options')
        datasetArgs.add_argument('--corpus', choices=TextData.corpusChoices(), default=TextData.corpusChoices()[0],
                                 help='corpus on which extract the dataset.')
        datasetArgs.add_argument('--datasetTag', type=str, default='',
                                 help='add a tag to the dataset (file where to load the vocabulary and the precomputed samples, not the original corpus). Useful to manage multiple versions')  # The samples are computed from the corpus if it does not exist already. There are saved in \'data/samples/\'
        datasetArgs.add_argument('--ratioDataset', type=float, default=1.0,
                                 help='ratio of dataset used to avoid using the whole dataset')  # Not implemented, useless ?
        datasetArgs.add_argument('--maxLength', type=int, default=10,
                                 help='maximum length of the sentence (for input and output), define number of maximum step of the RNN')
        datasetArgs.add_argument('--holdout', type=int, default=0,
                                 help='if 0, use whole data set as train set;'
                                      'if 1, use 3/4 data set as train set and 1/4 as test set;'
                                      'if 2, use 1/2 data set ad train set, 1/4 as development set and 1/4 as test set')
        # TODO 1: Implement holdout choice
        datasetArgs.add_argument('--historyInputs', action='store_true',
                                 help='if present, concatenate the previous 2 sentences as input, and next sentence as target')

        # Network options (Warning: if modifying something here, also make the change on save/loadParams() )
        nnArgs = parser.add_argument_group('Network options', 'architecture related option')
        nnArgs.add_argument('--hiddenSize', type=int, default=256, help='number of hidden units in each RNN cell')
        nnArgs.add_argument('--numLayers', type=int, default=2, help='number of rnn layers')
        nnArgs.add_argument('--embeddingSize', type=int, default=32, help='embedding size of the word representation')
        # TODO 2: Change fixed embedding to word2vec model with trained initial parameters
        nnArgs.add_argument('--initEmbeddings', action='store_true',
                            help='if present, the program will initialize the embeddings with pre-trained word2vec vectors')
        nnArgs.add_argument('--softmaxSamples', type=int, default=0,
                            help='Number of samples in the sampled softmax loss function. A value of 0 deactivates sampled softmax')
        nnArgs.add_argument('--useAttentions', action='store_true',
                            help='if present, the program will use attention mechanism')

        # Training options
        trainingArgs = parser.add_argument_group('Training options')
        trainingArgs.add_argument('--numEpochs', type=int, default=30, help='maximum number of epochs to run')
        trainingArgs.add_argument('--saveEvery', type=int, default=1400,
                                  help='nb of mini-batch step before creating a model checkpoint')
        trainingArgs.add_argument('--batchSize', type=int, default=10, help='mini-batch size')
        trainingArgs.add_argument('--learningRate', type=float, default=0.001, help='Learning rate')
        trainingArgs.add_argument('--dropout', type=float, default=0.9, help='Dropout rate (keep probabilities)')
        trainingArgs.add_argument('--mmiN', type=int, default=10, help='MMI model n-best parameter N')
        trainingArgs.add_argument('--maxGradientNorm', type=float, default=5.0,
                                  help='Clip gradients to this norm in SGD optimizer')
        trainingArgs.add_argument('--validate', type=int, default=0,
                                  help='if greater than 0, validating bleu score on validating samples with size --validate during training')

        trainingArgs.add_argument('--maxTurns', type=int, default=5, help='RL model max simulation turns')
        trainingArgs.add_argument('--discount', type=float, default=1, help='RL reward discount parameter')

        return parser.parse_args(args)

    def main(self, args=None):
        """
        Launch the training and/or the interactive mode
        """
        print('Welcome to DeepQA v0.1 !')
        print()
        print('TensorFlow detected: v{}'.format(tf.__version__))

        # General initialisation

        self.args = self.parseArgs(args)

        if not self.args.rootDir:
            self.args.rootDir = os.getcwd()  # Use the current working directory

        self.loadModelParams()

        self.textData = TextData(self.args)

        if self.args.createDataset:
            print('Dataset created! Thanks for using this program')
            return  # No need to go further

        with tf.device(self.getDevice()):
            # self.model = Model(self.args, self.textData)
            # 主模型（根据训练train选项分别代表seq2seq(fore or back),matual info,rl, test不同模型的支持还没有加)
            self.model = Model(self.args, self.textData)

            if self.args.train and self.args.test:
                self.args.test = False
            if self.args.train:
                self.args.test = True
                with tf.variable_scope("validation"):
                    self.vldmodel = Model(self.args, self.textData)
                with tf.variable_scope("forward"):
                    self.forward = Model(self.args, self.textData)
                with tf.variable_scope("backward"):
                    self.backward = Model(self.args, self.textData)
                self.args.test = False

        # Saver/summaries
        self.writer = tf.summary.FileWriter(self._getSummaryName())
        # tf0.12 and before:self.writer = tf.train.SummaryWriter(self._getSummaryName())

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.args.keepAll)

        # Running session
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        if self.args.train and self.args.validate > 0:
            self.vldsess = tf.Session(config=config)
        if self.args.train == Chatbot.TrainMode.MUTUALINFO or self.args.train == Chatbot.TrainMode.REINFORCEMENT:
            self.for2sess = tf.Session(config=config)
            self.backsess = tf.Session(config=config)

        print('Initialize variables...')
        self.sess.run(tf.global_variables_initializer())

        # Reload the model eventually (if it exist.), on testing mode,
        # the models are not loaded here (but in predictTestset)
        if self.args.test != Chatbot.TestMode.ALL:
            # 不使用--test参数时self.args.test为None，即所有train过程model restore均是在managePreciousModel中完成的
            self.managePreviousModel(self.sess)

        # Use attention mechanism in model by creating embedding_attention model
        if self.args.useAttentions:
            print('Using attention mechanism in model')
            if self.args.softmaxSamples == 0:
                print('Warning: Use attention mechanism without softmax samples '
                      'requires larger memory space and may raise OOM exception.')
                print('Recommend to rerun the program and train the model '
                      'with softmaxSamples and useAttentions arguments')

        # Initialize embeddings with pre-trained word2vec vectors
        if self.args.initEmbeddings:
            print("Loading pre-trained embeddings from GoogleNews-vectors-negative300.bin")
            self.loadEmbedding(self.sess)

        if self.args.test:
            if self.args.test == Chatbot.TestMode.INTERACTIVE:
                self.mainTestInteractive(self.sess)
            elif self.args.test == Chatbot.TestMode.ALL:
                print('Start predicting...')
                self.predictTestset(self.sess)
                print('All predictions done')
            elif self.args.test == Chatbot.TestMode.DAEMON:
                print('Daemon mode, running in background...')
            else:
                raise RuntimeError('Unknown test mode: {}'.format(self.args.test))  # Should never happen
        elif self.args.train:
            if self.args.train == Chatbot.TrainMode.SEQ2SEQ:
                self.mainTrain(self.sess)
            elif self.args.train == Chatbot.TrainMode.MUTUALINFO:
                self.miTrain(self.sess)
            elif self.args.train == Chatbot.TrainMode.REINFORCEMENT:
                self.rlTrain(self.sess)
            else:
                raise RuntimeError('Unknown train mode: {}'.format(self.args.train))  # Should never happen
        else:
            print(
                'Warning: Unknown program state, you need to use either --train or --test argument.')  # Should never happen

        if self.args.test != Chatbot.TestMode.DAEMON:
            self.sess.close()
            if self.args.train and self.args.validate > 0:
                self.vldsess.close()
            print("The End! Thanks for using this program")

    '''
    训练相关函数
    '''

    def mainTrain(self, sess):
        """ Training loop
        Args:
            sess: The current running session
        """

        # Specific training dependent loading

        self.textData.makeLighter(self.args.ratioDataset)  # Limit the number of training samples

        mergedSummaries = tf.summary.merge_all()
        if self.globStep == 0:  # Not restoring from previous run
            self.writer.add_graph(sess.graph)  # First time only

        # If restoring a model, restore the progression bar ? and current batch ?

        print('Start seq2seq model training (press Ctrl+C to save and exit)...')

        perfFile = open(os.path.join(self.modelDir, self.PERFORMANCE_FILENAME), 'w')

        try:  # If the user exit while training, we still try to save the model
            for e in range(self.args.numEpochs):

                print()
                print("----- Epoch {}/{} ; (lr={}) -----".format(e + 1, self.args.numEpochs, self.args.learningRate))
                perfFile.write("----- Epoch {}/{} -----".format(e + 1, self.args.numEpochs) + "\n")

                batches = self.textData.getBatches()

                # TODO 2: Also update learning parameters eventually

                tic = datetime.datetime.now()

                for nextBatch in tqdm(batches, desc="Training"):
                    # Training pass
                    ops, feedDict = self.model.step(nextBatch)

                    assert len(ops) == 2  # training, loss
                    _, loss, summary = sess.run(ops + (mergedSummaries,), feedDict)
                    self.writer.add_summary(summary, self.globStep)
                    self.globStep += 1

                    # Output training status
                    if self.globStep % 200 == 0:
                        perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                        tqdm.write("----- Step %d -- Loss %.2f -- Perplexity %.2f" % (self.globStep, loss, perplexity))
                        perfFile.write(
                            "----- Step %d -- Loss %.2f -- Perplexity %.2f\n" % (self.globStep, loss, perplexity))

                    # Checkpoint
                    if self.globStep % self.args.saveEvery == 0:
                        self._saveSession(sess)

                # *************************************************************************
                # Calculating BLEU score on random validation set with size k=validate
                if self.args.validate > 0:
                    self.args.test = True
                    vldBatches = self.textData.getBatches(validate=self.args.validate)
                    average_bleu = 0
                    refs = []
                    hyps = []
                    for vldbatch in tqdm(vldBatches, desc="Validating"):
                        assert len(vldbatch) == 2
                        inputSeqs = vldbatch[0]
                        targetSeqs = vldbatch[1]
                        question = self.textData.sequence2str(inputSeqs, clean=True)
                        questionSeq = []
                        print(question)
                        answer = self.singlePredict(question, questionSeq, vld=True)
                        ref = nltk.word_tokenize(self.textData.sequence2str(answer, clean=True))
                        print(ref)
                        refs.append([ref])
                        hyp = nltk.word_tokenize(self.textData.sequence2str(targetSeqs, clean=True))
                        print(hyp)
                        hyps.append(hyp)
                        bleu = bleu_score.sentence_bleu([ref], hyp,
                                                        smoothing_function=bleu_score.SmoothingFunction().method2,
                                                        weights=[0.3, 0.3, 0.2, 0.2])
                        print(bleu)
                        average_bleu += bleu
                    average_bleu /= len(self.textData.validatingSamples)
                    corpus_bleu = bleu_score.corpus_bleu(refs, hyps,
                                                         smoothing_function=bleu_score.SmoothingFunction().method2,
                                                         weights=[0.3, 0.3, 0.2, 0.2])
                    tqdm.write(
                        "----- Epoch %d -- Average BLEU %.4f -- Corpus BLEU %.4f" % (e + 1, average_bleu, corpus_bleu))
                    perfFile.write(
                        "----- Epoch %d -- Average BLEU %.4f -- Corpus BLEU %.4f\n" % (
                            e + 1, average_bleu, corpus_bleu))
                    self.args.test = False
                # ************************************************************************
                toc = datetime.datetime.now()

                print("Epoch finished in {}".format(
                    toc - tic))
                # Warning: Will overflow if an epoch takes more than 24 hours, and the output isn't really nicer
                perfFile.write("Epoch finished in {}".format(
                    toc - tic))
        except (KeyboardInterrupt, SystemExit):  # If the user press Ctrl+C while testing progress
            print('Interruption detected, exiting the program...')

        self._saveSession(sess)  # Ultimate saving before complete exit

        perfFile.flush()
        perfFile.close()

    def miTrain(self, sess):
        """ Training loop
        Args:
            sess: The current running session
        """
        self.textData.makeLighter(self.args.ratioDataset)  # Limit the number of training samples

        mergedSummaries = tf.summary.merge_all()

        # TODO 1:
        # if mode changed from seq2seq to mmi(maximum mutual information)
        #   self.globStep = 0
        #   writer.re_add_graph(sess.graph)
        # ps: sess.graph在model creation时已更改
        if self.globStep == 0:
            self.writer.add_graph(sess.graph)

        print('Start mutual information model training (press Ctrl+C to save and exit)...')
        try:
            for e in range(self.args.numEpochs):
                print()
                print("----- Epoch {}/{} ; (lr={}) -----".format(e + 1, self.args.numEpochs, self.args.learningRate))

                tic = datetime.datetime.now()

                # TODO 1: Implement Mutual Information model Training using mutual information score
                # Mutual information score definition(using in new object function)

                toc = datetime.datetime.now()

                print("Epoch finished in {}".format(
                    toc - tic))
        except (KeyboardInterrupt, SystemExit):
            print('Interruption detected, exiting the program...')

        self._saveSession(sess)

    def rlTrain(self, sess):
        """ Training loop
        Args:
            sess: The current running session
        """

        self.textData.makeLighter(self.args.ratioDataset)  # Limit the number of training samples

        mergedSummaries = tf.summary.merge_all()

        if self.globStep == 0:
            self.writer.add_graph(sess.graph)

        print('Start reinforcement training (press Ctrl+C to save and exit)...')
        try:
            modelPath1 = os.path.join(self.args.rootDir,
                                      self.MODEL_DIR_BASE) + '-' + Chatbot.ModelTag.SEQ2SEQ_TAG
            ckpt1 = tf.train.get_checkpoint_state(modelPath1)
            self.saver.restore(self.for2sess, ckpt1.model_checkpoint_path)
            modelPath2 = os.path.join(self.args.rootDir,
                                      self.MODEL_DIR_BASE) + '-' + Chatbot.ModelTag.BACKWARD_TAG
            ckpt2 = tf.train.get_checkpoint_state(modelPath2)
            self.saver.restore(self.backsess, ckpt2.model_checkpoint_path)

            def calculateReward(inputSeqs, outdist, outp, targetSeqs):
                '''

                :param inputSeqs: Batch
                :param targetSeqs: log(softmaxP) matrix
                :return: rewards on actions
                '''
                # Seq2Seq Model p(a|q)
                question = self.textData.sequence2str(inputSeqs, clean=True)
                questionSeq = []
                answer = self.singlePredict(question, questionSeq, rl=1)
                # Seq2Seq backward model p(q|a)
                backques = self.textData.sequence2str(targetSeqs, clean=True)
                backquesSeq = []
                origin = self.singlePredict(backques, backquesSeq, rl=2)

            for e in range(self.args.numEpochs):
                print()
                print("----- Epoch {}/{} ; (lr={}) -----".format(e + 1, self.args.numEpochs, self.args.learningRate))

                tic = datetime.datetime.now()

                datas = self.textData.getAgentMessages(size=1)
                if not self.args.historyInputs:

                    for data in tqdm(datas, desc="Reinforcement Training"):
                        paths = []

                        observations = []
                        actions = []
                        rewards = []

                        currentObserv = [Batch(), data]

                        def recurcive(currentObservation, t):
                            ops, feedDict = self.model.agentStep(currentObservation[1])

                            assert len(ops) == 1

                            actionP = [ops[0], feedDict]
                            output = sess.run(ops[0], feedDict)
                            py_ifx, seqences = self.textData.decoder2Nbest(output, self.args.mmiN)

                            # answer = self.textData.deco2sentence(output)
                            count = 0
                            for answer in seqences:
                                answerStr = self.textData.sequence2str(answer, clean=True)
                                answerBatch = self.textData.sentence2enco(answerStr)
                                action = [actionP, answerBatch]

                                nextObservation = [currentObservation[1], answerBatch]

                                reward = calculateReward(currentObservation[1], output, py_ifx[count], answer)

                                count += 1
                                observations.append(currentObservation)
                                actions.append(action)
                                rewards.append(reward)
                                if t < self.args.maxTurns:
                                    currentObservation = nextObservation
                                    recurcive(currentObservation, t + 1)
                                else:
                                    returnReward = []
                                    return_so_far = 0
                                    for t in range(len(rewards) - 1, -1, -1):
                                        return_so_far = rewards[t] + self.args.discount * return_so_far
                                        returnReward.append(return_so_far)
                                    # The returns are stored backwards in time, so we need to revert it
                                    returnReward = returnReward[::-1]

                                    # paths.append(dict(
                                    #     observations=np.array(observations),
                                    #     actions=np.array(actions),
                                    #     rewards=np.array(rewards),
                                    #     returns=np.array(returnReward)
                                    # ))
                                    paths.append(dict(
                                        observations=observations,
                                        actions=actions,
                                        rewards=np.array(rewards),
                                        returns=np.array(returnReward)
                                    ))
                                    observations.pop(len(observations)-1)
                                    actions.pop(len(actions)-1)
                                    rewards.pop(len(rewards)-1)
                                    returnReward.pop(len(returnReward)-1)
                            # observations.pop(len(observations) - 1)
                            # actions.pop(len(actions) - 1)
                            # rewards.pop(len(rewards) - 1)
                            # returnReward.pop(len(returnReward) - 1)

                        recurcive(currentObserv, t=1)



                        '''
                        定义returns_var tensor变量
                        '''
                        returns_var = tf.tensor('returns')
                        '''
                        初始化变量
                        '''
                        '''
                        形成其feed变量以执行后面的run
                        '''
                        feedReturns_var = []
                        # A=dist.log_likelihood_sym(actions_var, dist_info_vars)的含义：
                        # 即在状态下采取动作序列actions的概率，这个概率是需要第二次使用model foreward计算的
                        # 这一次foreward只需要定义每一次的forward操作，无需run，得到ops，里面的outputs即代表分布情况
                        # 之后与returns_var进行乘法运算(tensor运算）
                        # 需要记录每次forward的输入参数feedDict并形成整体feed变脸feedDictList
                        A = 0
                        feedDictList = []
                        surr = - tf.mean(A * returns_var)
                        self.model.mmiReward = surr
                        op = self.model.agentStep(batch=None, terminate=True)
                        sess.run(op, [feedDictList, feedReturns_var])
                else:
                    datas = []
                toc = datetime.datetime.now()

                print("Epoch finished in {}".format(
                    toc - tic))
        except (KeyboardInterrupt, SystemExit):
            print('Interruption detected, exiting the program...')

        self._saveSession(sess)

    def agentSimulation(self, sess, message):
        q = []
        p = []
        candidates = self.agentPredict(message, nbest=True)
        for i in range(self.args.maxTurns):
            p.append(candidates)
            qi = []
            for j in range(len(p[i])):
                qi.append(self.agentPredict(p[i][j]), nbest=True)
            q.append(qi)

        # TODO: calculate Rewards
        def computeMI(decoCan, encoIn, mmiParams=None):
            miscore = float()
            return miscore

        # TODO 1：直接继续定义lossfct2（应该说是改变里面的输入输出和weight参数指代，调用的可能还是tf.contrib.*）
        # TODO 1: 这里encoderInputs也要根据其数据结构进行处理，提取出相对应的完成onebatch语句
        i = 0
        convReward = [(tf.reduce_mean(
            computeMI(decoderCandidate, self.encoderInputs[i], self.mmiParams) for decoderCandidate in onebatch))
                      for onebatch in self.decoder2Nbest(self.outputs, self.args.mmiN)]
        # 一个batch的数据做一次综合Reward计算（类似上面的sequence loss吧虽然也不知道上面的sequence loss是不是这个意思）
        self.mmiReward = tf.reduce_mean(convReward)
        # TODO 1：修改summary的内容，summary应该是主要用于可视化的，具体作用不清楚，包括writer好像也是
        # 看一下scalar函数能不能重复赋值，是覆盖还是添加，想一下如果可视化的话在不同mode下是不是直接覆盖就可以，在上面扩大了graph的
        # 情况下，如果summary只添加一个loss function会不会不匹配而报错，或者直接先去掉可视化扩展相关的部分？

    '''
    测试相关函数
    '''

    def predictTestset(self, sess):
        """ Try predicting the sentences from the samples.txt file.
        The sentences are saved on the modelDir under the same name
        Args:
            sess: The current running session
        """

        # TODO 1: Add evaluation function, such as BLEU score, diversity and other evaluation number
        # TODO 1: Create reference answers of the sample file, which will be helpful in evaluation
        # TODO : 在history训练出的模型下（包括强化训练过程中也是使用history的），测试用例也需要处理出history用例，输入应为QA对，预测下一句，
        # 并调用historyPredict，而不是singlePredict（加分支判断load模型的historyInput参数）

        lines = []
        hypseqs = []
        refs = []
        hyps = []
        average_bleu = 0
        average_gleu = 0
        average_uni_ratio = 0
        average_bi_ratio = 0
        total_token = 0
        uni_dict = {}
        bi_dict = {}
        av_total_uni_ratio = 0.0
        av_total_bi_ratio = 0.0
        flag = self.textData.loadTestData(self.textData.testSamplesDir)
        if not flag:
            # Loading the file to predict
            with open(os.path.join(self.args.rootDir, self.TEST_IN_NAME), 'r') as f:
                lines = f.readlines()
        else:
            for sample in self.textData.testingSamples:
                lines.append(self.textData.sequence2str(sample[0], clean=True))
                hypseqs.append(self.textData.sequence2str(sample[1], clean=True))

        modelList = self._getModelList()
        if not modelList:
            print('Warning: No model found in \'{}\'. Please train a model before trying to predict'.format(
                self.modelDir))
            return

        # Predicting for each model present in modelDir
        for modelName in sorted(modelList):
            print('Restoring previous model from {}'.format(modelName))
            self.saver.restore(sess, modelName)
            print('Testing...')

            saveName = modelName[:-len(
                self.MODEL_EXT)] + self.TEST_OUT_SUFFIX  # We remove the model extension and add the prediction suffix
            with open(saveName, 'w') as f:
                nbIgnored = 0
                index = 0
                for line in tqdm(lines, desc='Sentences'):
                    if not flag:
                        question = line[:-1]  # Remove the endl character
                    else:
                        question = line

                    answer = self.singlePredict(question)
                    if not answer:
                        nbIgnored += 1
                        continue  # Back to the beginning, try again

                    if not flag:
                        predString = '{x[0]}{0}\n{x[1]}{1}\n\n'.format(question,
                                                                       self.textData.sequence2str(answer, clean=True),
                                                                       x=self.SENTENCES_PREFIX)
                    else:
                        predString = '{x[0]}{0}\n{x[1]}{1}\n{y}{2}\n\n'.format(question,
                                                                               self.textData.sequence2str(answer,
                                                                                                          clean=True),
                                                                               hypseqs[index],
                                                                               x=self.SENTENCES_PREFIX, y='T: ')

                        ref = nltk.word_tokenize(self.textData.sequence2str(answer, clean=True))
                        refs.append([ref])
                        hyp = nltk.word_tokenize(hypseqs[index])
                        hyps.append(hyp)

                        tokens = len(ref)
                        total_token += tokens
                        dic1 = {k: ref.count(k) for k in set(ref)}
                        uni_types = len(dic1)
                        dic2 = {}
                        for i in range(len(ref) - 1):
                            item = ref[i] + ' ' + ref[i + 1]
                            if item in dic2.keys():
                                dic2[item] += 1
                            else:
                                dic2[item] = 1
                        bi_types = len(dic2)
                        if tokens > 0:
                            uni_ratio = float(uni_types) / float(tokens)
                        else:
                            uni_ratio = 0
                        if tokens > 1:
                            bi_ratio = float(bi_types) / float(tokens-1)
                        else:
                            bi_ratio = 0
                        for it1 in dic1.keys():
                            if it1 in uni_dict.keys():
                                uni_dict[it1] += 1
                            else:
                                uni_dict[it1] = 1
                        for it2 in dic2.keys():
                            if it2 in bi_dict.keys():
                                bi_dict[it2] += 1
                            else:
                                bi_dict[it2] = 1

                        # bleu = bleu_score.sentence_bleu([ref], hyp, smoothing_function=bleu_score.SmoothingFunction().method2, weights=[0.3, 0.3, 0.2, 0.2])
                        bleu = bleu_score.sentence_bleu([ref], hyp,
                                                        smoothing_function=bleu_score.SmoothingFunction().method2)
                        try:
                            gleu = gleu_score.sentence_gleu(ref, hyp)
                        except (ZeroDivisionError):
                            print("Error: Division by zero, need smoothing function.")
                            gleu = 0.0
                        predString = predString + ("Sentence BLEU %.4f, Sentence Google-BlEU %.4f.\n"
                                                   "Unigram diversity %.4f, Bigram diversity %.4f.\n\n" % (
                                                       bleu, gleu, uni_ratio, bi_ratio))

                        average_bleu += bleu
                        average_gleu += gleu
                        average_bi_ratio += bi_ratio
                        average_uni_ratio += uni_ratio


                    if self.args.verbose:
                        tqdm.write(predString)
                    f.write(predString)
                    index += 1
                if flag:
                    average_bleu /= (len(lines) - nbIgnored)
                    average_gleu /= (len(lines) - nbIgnored)
                    average_uni_ratio /= (len(lines) - nbIgnored)
                    average_bi_ratio /= (len(lines) - nbIgnored)

                    av_total_uni_ratio = float(len(uni_dict)) / float(total_token)
                    av_total_bi_ratio = float(len(bi_dict)) / float(total_token - len(lines) + nbIgnored)
                    # corpus_bleu = bleu_score.corpus_bleu(refs, hyps,
                    #                                      smoothing_function=bleu_score.SmoothingFunction().method2,
                    #                                      weights=[0.3, 0.3, 0.2, 0.2])
                    corpus_bleu = bleu_score.corpus_bleu(refs, hyps,
                                                         smoothing_function=bleu_score.SmoothingFunction().method2)
                    f.write("Average BLEU %.4f, Average Google-BLEU %.4f, Corpus BLEU %.4f.\n"
                            "Average Unigram diversity %.4f, Average Bigram diversity %.4f.\n"
                            "Average T-Unigram diversity %.4f, Average T-Bigram diversity %.4f." % (
                                average_bleu, average_gleu, corpus_bleu, average_uni_ratio, average_bi_ratio,
                                av_total_uni_ratio, av_total_bi_ratio))
                print('Prediction finished, {}/{} sentences ignored (too long)'.format(nbIgnored, len(lines)))

    def mainTestInteractive(self, sess):
        """ Try predicting the sentences that the user will enter in the console
        Args:
            sess: The current running session
        """
        # TODO 2: Also show the top 10 most likely predictions for each predicted output (when verbose mode?)
        # TODO 3: Log the questions asked for latter re-use (merge with test/samples.txt)
        # TODO : 在history训练出的模型下（包括强化训练过程中也是使用history的），interactive测试也需要处理出history信息，
        # 并调用historyPredict，而不是singlePredict（加分支判断load模型的historyInput参数）

        print('Testing: Launch interactive mode:')
        print('')
        print('Welcome to the interactive mode, here you can ask to Deep Q&A the sentence you want. Don\'t have high '
              'expectation. Type \'exit\' or just press ENTER to quit the program. Have fun.')

        while True:
            question = input(self.SENTENCES_PREFIX[0])
            if question == '' or question == 'exit':
                break

            questionSeq = []  # Will be contain the question as seen by the encoder
            answer = self.singlePredict(question, questionSeq)
            if not answer:
                print('Warning: sentence too long, sorry. Maybe try a simpler sentence.')
                continue  # Back to the beginning, try again

            print('{}{}'.format(self.SENTENCES_PREFIX[1], self.textData.sequence2str(answer, clean=True)))

            if self.args.verbose:
                print(self.textData.batchSeq2str(questionSeq, clean=True, reverse=True))
                print(self.textData.sequence2str(answer))

            print()

    def singlePredict(self, question, questionSeq=None, vld=False, rl=0, nbest=False):
        """ Predict the sentence
        Args:
            question (str): the raw input sentence
            questionSeq (List<int>): output argument. If given will contain the input batch sequence
        Return:
            list <int>: the word ids corresponding to the answer
        """
        # Create the input batch
        batch = self.textData.sentence2enco(question)
        if not batch:
            return None
        if questionSeq is not None:  # If the caller want to have the real input
            questionSeq.extend(batch.encoderSeqs)

        # Run the model
        ops = None
        output = None
        if not vld and rl == 0:
            ops, feedDict = self.model.step(batch)
            output = self.sess.run(ops[0], feedDict)
        elif vld:
            ckpt = tf.train.get_checkpoint_state(self.modelDir)
            self.saver.restore(self.vldsess, ckpt.model_checkpoint_path)
            ops, feedDict = self.vldmodel.step(batch)
            output = self.vldsess.run(ops[0], feedDict)
        elif rl == 1:
            ops, feedDict = self.model.step(batch)
            output = self.for2sess.run(ops[0], feedDict)
        elif rl == 2:
            ops, feedDict = self.backward.step(batch)
            output = self.backsess.run(ops[0], feedDict)
        if not nbest:
            answer = self.textData.deco2sentence(output)
        else:
            candidates = self.textData.decoder2Nbest(decoderOuts=output, N=self.args.mmiN)
            answer = []
            for can in candidates:
                answer.append(self.textData.deco2sentence(can))
        # answer = self.textData.deco2sentence(output)
        return answer

    def daemonPredict(self, sentence):
        """ Return the answer to a given sentence (same as singlePredict() but with additional cleaning)
        Args:
            sentence (str): the raw input sentence
        Return:
            str: the human readable sentence
        """
        # daemon模式是后台运行模式，基本和interactive模式一致，不过是运行在server端，为客户端应用（如web）提供计算服务
        # TODO : 在history训练出的模型下（包括强化训练过程中也是使用history的），daemon测试也需要处理出history信息，
        # 并调用historyPredict，而不是singlePredict（在加载模型时检查historyInput参数，加分支判断）
        return self.textData.sequence2str(
            self.singlePredict(sentence),
            clean=True
        )

    # def agentsTestInteractive(self, sess):

    def daemonClose(self):
        """ A utility function to close the daemon when finish
        """
        print('Exiting the daemon mode...')
        self.sess.close()
        print('Daemon closed.')

    def agentPredict(self, question, questionSeq=None, vld=False, rl=0, nbest=False):
        """ Predict the sentence
        Args:
            question (str): the raw input sentence
            questionSeq (List<int>): output argument. If given will contain the input batch sequence
        Return:
            list <int>: the word ids corresponding to the answer
        """
        # TODO 1: Implement history predict function
        # Create the input batch
        batch = self.textData.sentence2enco(question, ishistory=True)
        if not batch:
            return None
        if questionSeq is not None:  # If the caller want to have the real input
            questionSeq.extend(batch.encoderSeqs)

        # Run the model
        ops = None
        output = None
        if not vld and rl == 0:
            ops, feedDict = self.model.step(batch)
            output = self.sess.run(ops[0], feedDict)
            print(output)
        elif vld:
            ckpt = tf.train.get_checkpoint_state(self.modelDir)
            self.saver.restore(self.vldsess, ckpt.model_checkpoint_path)
            ops, feedDict = self.vldmodel.step(batch)
            output = self.vldsess.run(ops[0], feedDict)
        elif rl == 1:
            ops, feedDict = self.model.step(batch)
            output = self.for2sess.run(ops[0], feedDict)
        elif rl == 2:
            ops, feedDict = self.backward.step(batch)
            output = self.backsess.run(ops[0], feedDict)
        if not nbest:
            answer = self.textData.deco2sentence(output)
        else:
            candidates = self.textData.decoder2Nbest(decoderOuts=output, N=self.args.mmiN)
            answer = []
            for can in candidates:
                answer.append(self.textData.deco2sentence(can))
        return answer

    # def evaluations(self):
    #     """ An evaluation function to judge the performance of the model on test set
    #     :return:
    #         score: the quantized score of model performance
    #     """
    #     # TODO 1: Implement evaluations on test set
    #     # BLEU score
    #     # Perplexity
    #     # Length of the dialogue->only use in agents performance evaluations
    #     # Diversity:type-token ratio for unigrams and bigrams(seq2seq,rl->beam search(10),mutual information->n-best list(10) use only in testing)
    #     # Human evaluation
    #     score = 0
    #     return score

    '''
    辅助函数
    '''

    def loadEmbedding(self, sess):
        """ Initialize embeddings with pre-trained word2vec vectors
        Will modify the embedding weights of the current loaded model
        Uses the GoogleNews pre-trained values (path hardcoded)
        """

        # Fetch embedding variables from model
        if self.args.useAttentions:
            with tf.variable_scope("embedding_attention_seq2seq/rnn/embedding_wrapper", reuse=True):
                em_in = tf.get_variable("embedding")
            with tf.variable_scope("embedding_attention_seq2seq/embedding_attention_decoder", reuse=True):
                em_out = tf.get_variable("embedding")
        else:
            with tf.variable_scope("embedding_rnn_seq2seq/rnn/embedding_wrapper", reuse=True):
                em_in = tf.get_variable("embedding")
            with tf.variable_scope("embedding_rnn_seq2seq/embedding_rnn_decoder", reuse=True):
                em_out = tf.get_variable("embedding")

        # Disable training for embeddings
        variables = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
        variables.remove(em_in)
        variables.remove(em_out)

        # If restoring a model, we can leave here
        if self.globStep != 0:
            return

        # New model, we load the pre-trained word2vec data and initialize embeddings
        with open(os.path.join(self.args.rootDir, 'data/word2vec/GoogleNews-vectors-negative300.bin'), "rb", 0) as f:
            header = f.readline()
            vocab_size, vector_size = map(int, header.split())
            binary_len = np.dtype('float32').itemsize * vector_size
            initW = np.random.uniform(-0.25, 0.25, (len(self.textData.word2id), vector_size))
            for line in tqdm(range(vocab_size)):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        word = b''.join(word).decode('utf-8')
                        break
                    if ch != b'\n':
                        word.append(ch)
                if word in self.textData.word2id:
                    initW[self.textData.word2id[word]] = np.fromstring(f.read(binary_len), dtype='float32')
                else:
                    f.read(binary_len)

        # PCA Decomposition to reduce word2vec dimensionality
        if self.args.embeddingSize < vector_size:
            U, s, Vt = np.linalg.svd(initW, full_matrices=False)
            S = np.zeros((vector_size, vector_size), dtype=complex)
            S[:vector_size, :vector_size] = np.diag(s)
            initW = np.dot(U[:, :self.args.embeddingSize], S[:self.args.embeddingSize, :self.args.embeddingSize])

        # Initialize input and output embeddings
        sess.run(em_in.assign(initW))
        sess.run(em_out.assign(initW))

    def managePreviousModel(self, sess):
        """ Restore or reset the model, depending of the parameters
        If the destination directory already contains some file, it will handle the conflict as following:
         * If --reset is set, all present files will be removed (warning: no confirmation is asked) and the training
         restart from scratch (globStep & cie reinitialized)
         * Otherwise, it will depend of the directory content. If the directory contains:
           * No model files (only summary logs): works as a reset (restart from scratch)
           * Other model files, but modelName not found (surely keepAll option changed): raise error, the user should
           decide by himself what to do
           * The right model file (eventually some other): no problem, simply resume the training
        In any case, the directory will exist as it has been created by the summary writer
        Args:
            sess: The current running session
        """

        # print('WARNING: ', end='')

        modelName = self._getModelName()

        ckpt = tf.train.get_checkpoint_state(self.modelDir)

        if os.listdir(self.modelDir):
            if self.args.reset:
                print('Reset: Destroying previous model at {}'.format(self.modelDir))
            # Analysing directory content
            elif ckpt and tf.train.checkpoint_exists(
                    ckpt.model_checkpoint_path) and modelName + '-' + str(self.globStep) == ckpt.model_checkpoint_path:
                # os.path.exists(modelName):  # Restore the model
                print('Restoring previous model from {}'.format(modelName + '-' + str(self.globStep)))
                self.saver.restore(sess, ckpt.model_checkpoint_path)
                # self.saver.restore(sess, modelName)
                # Will crash when --reset is not activated and the model has not been saved yet
                print('Model restored.')
            elif self._getModelList():
                print(self._getModelList())
                print('Conflict with previous models.')
                raise RuntimeError(
                    'Some models are already present in \'{}\'. You should check them first.'.format(
                        self.modelDir))
            else:  # No other model to conflict with (probably summary files)
                print('No previous model found, but some files found at {}. Cleaning...'.format(
                    self.modelDir))  # Warning: No confirmation asked
                self.args.reset = True

            if self.args.reset:
                fileList = [os.path.join(self.modelDir, f) for f in os.listdir(self.modelDir)]
                for f in fileList:
                    print('Removing {}'.format(f))
                    os.remove(f)

        else:
            print('No previous model found, starting from clean directory: {}'.format(self.modelDir))

    def _saveSession(self, sess):
        """ Save the model parameters and the variables
        Args:
            sess: the current session
        """
        tqdm.write('Checkpoint reached: saving model (don\'t stop the run)...')
        self.saveModelParams()
        self.saver.save(sess, self._getModelName(), global_step=self.globStep)
        tqdm.write('Model saved.')

    def _getModelList(self):
        """ Return the list of the model files inside the model directory
        """
        return [os.path.join(self.modelDir, f[0:f.index('.index')]) for f in os.listdir(self.modelDir) if
                f.endswith('.index')]

    def loadModelParams(self):
        """ Load the some values associated with the current model, like the current globStep value
        For now, this function does not need to be called before loading the model (no parameters restored). However,
        the modelDir name will be initialized here so it is required to call this function before managePreviousModel(),
        _getModelName() or _getSummaryName()
        Warning: if you modify this function, make sure the changes mirror saveModelParams, also check if the parameters
        should be reset in managePreviousModel
        """
        # Compute the current model path
        self.modelDir = os.path.join(self.args.rootDir, self.MODEL_DIR_BASE)
        if self.args.modelTag:
            self.modelDir += '-' + self.args.modelTag

        # If there is a previous model, restore some parameters
        configName = os.path.join(self.modelDir, self.CONFIG_FILENAME)
        if not self.args.reset and not self.args.createDataset and os.path.exists(configName):
            # Loading
            config = configparser.ConfigParser()
            config.read(configName)

            # Check the version
            currentVersion = config['General'].get('version')
            if currentVersion != self.CONFIG_VERSION:
                raise UserWarning(
                    'Present configuration version {0} does not match {1}. You can try manual changes on \'{2}\''.format(
                        currentVersion, self.CONFIG_VERSION, configName))

            # Restoring the the parameters
            self.globStep = config['General'].getint('globStep')
            self.args.maxLength = config['General'].getint(
                'maxLength')
            # We need to restore the model length because of the textData associated and the vocabulary size
            # TODO 3: Compatibility mode between different maxLength
            self.args.watsonMode = config['General'].getboolean('watsonMode')
            self.args.corpus = config['General'].get('corpus')
            self.args.datasetTag = config['General'].get('datasetTag', '')
            self.args.holdout = config['General'].getint('holdout')
            self.args.historyInputs = config['General'].getboolean('historyInputs')

            self.args.hiddenSize = config['Network'].getint('hiddenSize')
            self.args.numLayers = config['Network'].getint('numLayers')
            self.args.embeddingSize = config['Network'].getint('embeddingSize')
            self.args.initEmbeddings = config['Network'].getboolean('initEmbeddings')
            self.args.softmaxSamples = config['Network'].getint('softmaxSamples')
            self.args.useAttentions = config['Network'].getboolean('useAttentions')

            # No restoring for training params, batch size or other non model dependent parameters

            # Show the restored params
            print()
            print('Warning: Restoring parameters:')
            print('globStep: {}'.format(self.globStep))
            print('maxLength: {}'.format(self.args.maxLength))
            print('watsonMode: {}'.format(self.args.watsonMode))
            print('corpus: {}'.format(self.args.corpus))
            print('datasetTag: {}'.format(self.args.datasetTag))
            print('holdout: {}'.format(self.args.holdout))
            print('historyInputs: {}'.format(self.args.historyInputs))
            print('hiddenSize: {}'.format(self.args.hiddenSize))
            print('numLayers: {}'.format(self.args.numLayers))
            print('embeddingSize: {}'.format(self.args.embeddingSize))
            print('initEmbeddings: {}'.format(self.args.initEmbeddings))
            print('softmaxSamples: {}'.format(self.args.softmaxSamples))
            print('useAttentions: {}'.format(self.args.useAttentions))
            print()

        # For now, not arbitrary  independent maxLength between encoder and decoder
        # h&w: ((pi,qi)|a)
        # not h&not w: (a|pi)
        # ** not h&w: (qi|a)
        # ** h&not w: (a|(qi,pi))
        if not self.args.watsonMode:
            if not self.args.historyInputs:
                self.args.maxLengthEnco = self.args.maxLength
            else:
                self.args.maxLengthEnco = 2 * self.args.maxLength
            self.args.maxLengthDeco = self.args.maxLength + 2
        else:
            self.args.maxLengthEnco = self.args.maxLength
            if not self.args.historyInputs:
                self.args.maxLengthDeco = self.args.maxLength + 2
            else:
                self.args.maxLengthDeco = 2 * self.args.maxLength + 2

        if self.args.watsonMode:
            self.SENTENCES_PREFIX.reverse()

    def saveModelParams(self):
        """ Save the params of the model, like the current globStep value
        Warning: if you modify this function, make sure the changes mirror loadModelParams
        """
        config = configparser.ConfigParser()
        config['General'] = {}
        config['General']['version'] = self.CONFIG_VERSION
        config['General']['globStep'] = str(self.globStep)
        config['General']['maxLength'] = str(self.args.maxLength)
        config['General']['watsonMode'] = str(self.args.watsonMode)
        config['General']['corpus'] = str(self.args.corpus)
        config['General']['datasetTag'] = str(self.args.datasetTag)
        config['General']['holdout'] = str(self.args.holdout)
        config['General']['historyInputs'] = str(self.args.historyInputs)

        config['Network'] = {}
        config['Network']['hiddenSize'] = str(self.args.hiddenSize)
        config['Network']['numLayers'] = str(self.args.numLayers)
        config['Network']['embeddingSize'] = str(self.args.embeddingSize)
        config['Network']['initEmbeddings'] = str(self.args.initEmbeddings)
        config['Network']['softmaxSamples'] = str(self.args.softmaxSamples)
        config['Network']['useAttentions'] = str(self.args.useAttentions)

        # Keep track of the learning params (but without restoring them)
        config['Training (won\'t be restored)'] = {}
        config['Training (won\'t be restored)']['learningRate'] = str(self.args.learningRate)
        config['Training (won\'t be restored)']['batchSize'] = str(self.args.batchSize)
        config['Training (won\'t be restored)']['dropout'] = str(self.args.dropout)
        config['Training (won\'t be restored)']['mmiN'] = str(self.args.mmiN)
        config['Training (won\'t be restored)']['maxGradientNorm'] = str(self.args.maxGradientNorm)
        config['Training (won\'t be restored)']['maxTurns'] = str(self.args.maxTurns)
        config['Training (won\'t be restored)']['discount'] = str(self.args.discount)
        config['Training (won\'t be restored)']['validate'] = str(self.args.validate)

        with open(os.path.join(self.modelDir, self.CONFIG_FILENAME), 'w') as configFile:
            config.write(configFile)

    def _getSummaryName(self):
        """ Parse the argument to decide were to save the summary, at the same place that the model
        The folder could already contain logs if we restore the training, those will be merged
        Return:
            str: The path and name of the summary
        """
        return self.modelDir

    def _getModelName(self):
        """ Parse the argument to decide were to save/load the model
        This function is called at each checkpoint and the first time the model is load. If keepAll option is set, the
        globStep value will be included in the name.
        Return:
            str: The path and name were the model need to be saved
        """
        modelName = os.path.join(self.modelDir, self.MODEL_NAME_BASE)
        # if self.args.keepAll:  # We do not erase the previously saved model by including the current step on the name
        #     modelName += '-' + str(self.globStep)
        return modelName + self.MODEL_EXT

    def getDevice(self):
        """ Parse the argument to decide on which device run the model
        Return:
            str: The name of the device on which run the program
        """
        if self.args.device == 'cpu':
            return '/cpu:0'
        elif self.args.device == 'gpu0' or self.args.device == 'gpu':
            return '/gpu:0'
        elif self.args.device == 'gpu1':
            return '/gpu:1'
        elif self.args.device == 'gpu2':
            return '/gpu:2'
        elif self.args.device is None:  # No specified device (default)
            return None
        else:
            print('Warning: Error in the device name: {}, use the default device'.format(self.args.device))
            return None
