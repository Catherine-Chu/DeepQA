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
Model to predict the next sentence given an input sequence

"""

import tensorflow as tf
import numpy as np
import heapq

from chatbot.textdata import Batch


class ProjectionOp:
    """ Single layer perceptron
    Project input tensor on the output dimension
    """

    def __init__(self, shape, scope=None, dtype=None):
        """
        Args:
            shape: a tuple (input dim, output dim)
            scope (str): encapsulate variables
            dtype: the weights type
        """
        assert len(shape) == 2

        self.scope = scope

        # Projection on the keyboard
        with tf.variable_scope('weights_' + self.scope):
            self.W = tf.get_variable(
                'weights',
                shape,
                dtype=dtype
            )
            self.b = tf.get_variable(
                'bias',
                shape[1],
                initializer=tf.constant_initializer(),
                dtype=dtype
            )

    def getWeights(self):
        """ Convenience method for some tf arguments
        """
        return self.W, self.b

    def __call__(self, X):
        """ Project the output of the decoder into the vocabulary space
        Args:
            X (tf.Tensor): input value
        """
        with tf.name_scope(self.scope):
            return tf.matmul(X, self.W) + self.b


class Model:
    """
    Implementation of a seq2seq model.
    Achitecture:
        2 LTSM layers
    """

    def __init__(self, args, textData):
        """
        Args:
            args: parametters of the model
            textData: the dataset object
        """
        print("Model creation...")

        self.textData = textData  # Keep a reference on the dataset
        self.args = args  # Keep track of the parameters of the model
        self.dtype = tf.float32

        # Placeholders
        self.encoderInputs = None
        self.decoderInputs = None  # Same that decoderTarget plus the <go>
        self.decoderTargets = None
        self.decoderWeights = None  # Adjust the learning to the target sentence size
        self.mmiParams = None

        # Main operators
        self.lossFct = None
        self.optOp = None
        self.mmiOp = None
        self.mmiReward = None
        self.outputs = None  # Outputs of the network, list of probability for each words

        # SGD optimizer track
        self.gradient_norms = []
        self.updates = []

        # Construct the graphs
        self.buildNetwork()

    def buildNetwork(self):
        """ Create the computational graph
        """

        # TODO 3: Create name_scopes (for better graph visualisation)
        # TODO 2: Use buckets (better perfs), learning the short/easy sentence first, with shuffling inside buckets instead of global

        # Parameters of sampled softmax (needed for attention mechanism and a large vocabulary size)
        outputProjection = None
        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if 0 < self.args.softmaxSamples < self.textData.getVocabularySize():
            outputProjection = ProjectionOp(
                (self.args.hiddenSize, self.textData.getVocabularySize()),
                scope='softmax_projection',
                dtype=self.dtype
            )

            def sampledSoftmax(labels, inputs):
                labels = tf.reshape(labels, [-1, 1])  # Add one dimension (nb of true classes, here 1)

                # We need to compute the sampled_softmax_loss using 32bit floats to
                # avoid numerical instabilities.
                localWt = tf.cast(tf.transpose(outputProjection.W), tf.float32)
                localB = tf.cast(outputProjection.b, tf.float32)
                localInputs = tf.cast(inputs, tf.float32)

                return tf.cast(
                    tf.nn.sampled_softmax_loss(
                        localWt,  # Should have shape [num_classes, dim] 从更大维度映射到指定维度，即从整体词集映射到sample大小词集
                        localB,   # [dim]
                        labels,
                        localInputs,
                        self.args.softmaxSamples,  # The number of classes to randomly sample per batch
                        self.textData.getVocabularySize()),  # The number of classes
                    self.dtype)

        # Creation of the rnn cell
        def create_rnn_cell():
            encoDecoCell = tf.contrib.rnn.BasicLSTMCell(  # Or GRUCell, LSTMCell(args.hiddenSize)
                self.args.hiddenSize,
            )
            if not self.args.test:  # TODO 3: Should use a placeholder instead
                encoDecoCell = tf.contrib.rnn.DropoutWrapper(
                    encoDecoCell,
                    input_keep_prob=1.0,
                    output_keep_prob=self.args.dropout
                )
            return encoDecoCell

        encoDecoCell = tf.contrib.rnn.MultiRNNCell(
            [create_rnn_cell() for _ in range(self.args.numLayers)],
        )

        # Network input (placeholders)

        with tf.name_scope('placeholder_encoder'):
            self.encoderInputs = [tf.placeholder(tf.int32, [None, ]) for _ in
                                  range(self.args.maxLengthEnco)]  # Batch size * sequence length * input dim


        with tf.name_scope('placeholder_decoder'):
            self.decoderInputs = [tf.placeholder(tf.int32, [None, ], name='inputs') for _ in
                                  range(self.args.maxLengthDeco)]  # Same sentence length for input and output (Right ?)
            self.decoderTargets = [tf.placeholder(tf.int32, [None, ], name='targets') for _ in
                                   range(self.args.maxLengthDeco)]
            self.decoderWeights = [tf.placeholder(tf.float32, [None, ], name='weights') for _ in
                                   range(self.args.maxLengthDeco)]
        with tf.name_scope('mmi_parameters'):
            self.mmiParams = tf.Variable([1.0, 5.0], name="mmi_params", dtype=tf.float32)


        if not self.args.useAttentions:
            decoderOutputs, states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
                self.encoderInputs,  # List<[batch=?, inputDim=1]>, list of size args.maxLength
                self.decoderInputs,  # For training, we force the correct output (feed_previous=False)
                encoDecoCell,
                num_encoder_symbols=self.textData.getVocabularySize(),
                num_decoder_symbols=self.textData.getVocabularySize(),
                # Both encoder and decoder have the same number of class
                embedding_size=self.args.embeddingSize,  # Dimension of each word
                output_projection=outputProjection.getWeights() if outputProjection else None,
                feed_previous=bool(self.args.test)
                # When we test (self.args.test), we use previous output as next input (feed_previous)
            )
        else:
            decoderOutputs, states = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                self.encoderInputs,
                self.decoderInputs,
                encoDecoCell,
                num_encoder_symbols=self.textData.getVocabularySize(),
                num_decoder_symbols=self.textData.getVocabularySize(),
                embedding_size=self.args.embeddingSize,
                output_projection=outputProjection.getWeights() if outputProjection else None,
                feed_previous=bool(self.args.test),
                dtype=self.dtype
            )

        # TODO 3: When the LSTM hidden size is too big, we should project the LSTM output into a smaller space (4086 => 2046)
        # Purpose: Should speed up training and reduce memory usage
        # Other solution, use sampling softmax For testing only
        if self.args.test:
            if not outputProjection:
                self.outputs = decoderOutputs
            else:
                self.outputs = [outputProjection(output) for output in decoderOutputs]
                # TODO 3: Attach a summary to visualize the output
        # For training only
        else:
            # Finally, we define the loss function

            self.lossFct = tf.contrib.legacy_seq2seq.sequence_loss(
                decoderOutputs,
                self.decoderTargets,
                self.decoderWeights,
                self.textData.getVocabularySize(),
                softmax_loss_function=sampledSoftmax if outputProjection else None  # If None, use default SoftMax
            )

            tf.summary.scalar('loss', self.lossFct)  # Keep track of the cost

            # Initialize the optimizer
            opt = tf.train.AdamOptimizer(
                # 此处使用的优化算法与官方模型中不同，官方模型使用随机梯度下降方法GradientDescentOptimizer
                # Adam相较于SGD是更优的优化器
                learning_rate=self.args.learningRate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-08
            )

            self.optOp = opt.minimize(self.lossFct)

            if self.args.train == 'mutualinfo' or self.args.train == 'reinforcement':
                def computeMI(decoCan, encoIn, mmiParams=None):
                    miscore = float()
                    return miscore

                if not outputProjection:
                    self.outputs = decoderOutputs
                else:
                    self.outputs = [outputProjection(output) for output in decoderOutputs]

                # TODO 1：直接继续定义lossfct2（应该说是改变里面的输入输出和weight参数指代，调用的可能还是tf.contrib.*）
                # TODO 1: 这里encoderInputs也要根据其数据结构进行处理，提取出相对应的完成onebatch语句
                i = 0
                convReward = [(tf.reduce_mean(computeMI(decoderCandidate, self.encoderInputs[i], self.mmiParams) for decoderCandidate in onebatch))
                              for onebatch in self.decoder2Nbest(self.outputs, self.args.mmiN)]
                # 一个batch的数据做一次综合Reward计算（类似上面的sequence loss吧虽然也不知道上面的sequence loss是不是这个意思）
                self.mmiReward = tf.reduce_mean(convReward)
                # TODO 1：修改summary的内容，summary应该是主要用于可视化的，具体作用不清楚，包括writer好像也是
                # 看一下scalar函数能不能重复赋值，是覆盖还是添加，想一下如果可视化的话在不同mode下是不是直接覆盖就可以，在上面扩大了graph的
                # 情况下，如果summary只添加一个loss function会不会不匹配而报错，或者直接先去掉可视化扩展相关的部分？
                tf.summary.scalar('mmi_reward', self.mmiReward)

                params = tf.trainable_variables()
                opt1 = tf.train.GradientDescentOptimizer(
                    self.args.learningRate
                )
                gradients = tf.gradients(self.mmiReward, params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                                 self.args.maxGradientNorm)
                self.gradient_norms.append(norm)
                self.updates.append(opt1.apply_gradients(
                    zip(clipped_gradients, params)))

                self.mmiOp = opt1.minimize(-self.mmiReward)

    def step(self, batch):
        """ Forward/training step operation.
        Does not perform run on itself but just return the operators to do so. Those have then to be run
        Args:
            batch (Batch): Input data on testing mode, input and target on output mode
            vld: validation flag
        Return:
            (ops), dict: A tuple of the (training, loss) operators or (outputs,) in testing mode with the associated feed dictionary
        """

        # Feed the dictionary
        feedDict = {}
        ops = None

        if not self.args.test:  # Training
            for i in range(self.args.maxLengthEnco):
                feedDict[self.encoderInputs[i]] = batch.encoderSeqs[i]
            for i in range(self.args.maxLengthDeco):
                feedDict[self.decoderInputs[i]] = batch.decoderSeqs[i]
                feedDict[self.decoderTargets[i]] = batch.targetSeqs[i]
                feedDict[self.decoderWeights[i]] = batch.weights[i]
            if not self.args.train == 'matualinfo':
                ops = (self.optOp, self.lossFct)
            else:
                ops = (self.outputs, self.mmiOp, self.mmiReward)
        else:  # Testing (batchSize == 1)

            for i in range(self.args.maxLengthEnco):
                feedDict[self.encoderInputs[i]] = batch.encoderSeqs[i]
            feedDict[self.decoderInputs[0]] = [self.textData.goToken]

            ops = (self.outputs,)

        # Return one pass operator
        return ops, feedDict

    def agentStep(self, batch, turns):
        # TODO: 一组训练(5 turns)中把每个step返回的reward累加，形成新的reward赋值给model.mmiReward变量
        # 之后只需要做一次backward训练，调用最后一个step中返回的self.mmiOp进行Optimizer
        # 先实现r1与r3，r2这种涉及多轮的reward暂时不做
        # 可能可以实现r2的方法是在step外，根据返回的outputs计算r2，并最后加在reward上反馈回来
        # 要在Model中直接定义比较复杂，可能需要agent1H，和agent2H两个list变量记录5 turns历史，考虑到训练形式可能list还不够？
        # 保存output值并定义相似度？
        # Feed the dictionary
        feedDict = {}
        ops = None

        if not self.args.test and (self.args.train == 'matualinfo' or self.args.train == 'reinforcement'):  # Training
            for i in range(self.args.maxLengthEnco):
                feedDict[self.encoderInputs[i]] = batch.encoderSeqs[i]

            feedDict[self.decoderInputs[0]] = [self.textData.goToken]

            for i in range(self.args.maxLengthDeco):
                feedDict[self.decoderWeights[i]] = batch.weights[i]

                ops = (self.outputs, self.mmiOp, self.mmiReward)
        else:  # Testing (batchSize == 1)

            for i in range(self.args.maxLengthEnco):
                feedDict[self.encoderInputs[i]] = batch.encoderSeqs[i]
            feedDict[self.decoderInputs[0]] = [self.textData.goToken]

            ops = (self.outputs,)

        # Return one pass operator
        return ops, feedDict

    def decoder2Nbest(self, decoderOuts, N, fore2back=False):
        """Decode the output of the decoder and return a human friendly sentence
                decoderOutputs (list<np.array>):代表一句(fore2back=true)或多句输出各个单词组成的list，
                list中每个元素是一个(fore2back=true)或多个np.array，代表该预测位置在整个词表中的每个词可能出现的概率
        """

        def step(decoderOutputs):
            sequences = []
            n_best_w = []
            n_best_logit = []
            for out in decoderOutputs:
                n_best_w.append(heapq.nlargest(N, range(len(out)), out.take))
                # n_best_logit.append([out[x] for x in n_best_w[len(n_best_w) - 1]])
                n_best_logit.append(heapq.nlargest(N, out))

            # print(n_best_logit)
            logits = [1]
            paths = [list()]
            pos = 0
            for candidate_words in n_best_logit:
                logits1 = []
                paths1 = []
                for ws in range(len(candidate_words)):
                    for mul in range(len(logits)):
                        logits1.append(candidate_words[ws] * logits[mul])
                        a = []
                        a.extend(paths[mul])
                        a.append(n_best_w[pos][ws])
                        paths1.append(a)
                        # a.clear()
                logits.clear()
                paths.clear()
                if len(logits1) > 5000:
                    k1 = heapq.nlargest(5000, range(len(logits1)), np.array(logits1).take)
                    for ti in k1:
                        logits.append(logits1[ti])
                        paths.append(paths1[ti])
                else:
                    logits.extend(logits1)
                    paths.extend(paths1)
                logits1.clear()
                paths1.clear()
                pos += 1
            k = heapq.nlargest(N, range(len(logits)), np.array(logits).take)
            logitsseqs = heapq.nlargest(N, logits)
            for t in k:
                sequences.append(paths[t])
            return logitsseqs, sequences

        batchSeqs = []
        Py_ifx = []
        if not fore2back:
            decoderOuts = self.splitOutBatches(decoderOuts)
            for decoderOutputs in decoderOuts:
                logitsseqs, sequences = step(decoderOutputs)
                Py_ifx.append(logitsseqs)
                batchSeqs.append(sequences)
        else:
            logitsseqs, sequences = step(decoderOuts)
            Py_ifx.extend(logitsseqs)
            batchSeqs.extend(sequences)
        return Py_ifx, batchSeqs

    def splitOutBatches(self, decoderOuts):
        seqs = []
        w = 0
        for eachwords in decoderOuts:
            for eachone in range(self.args.batchSize):
                if w == 0:
                    seqs.append([])
                seqs[eachone].append(np.array(eachwords[eachone]))
            w += 1
        return seqs

