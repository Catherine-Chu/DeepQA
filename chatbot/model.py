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
                # initializer=tf.truncated_normal_initializer()  # TODO: Tune value (fct of input size: 1/sqrt(input_dim))
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

        # Main operators
        self.lossFct = None
        self.optOp = None
        self.outputs = None  # Outputs of the network, list of probability for each words

        # Construct the graphs
        self.buildNetwork()

    def buildNetwork(self):
        """ Create the computational graph
        """

        # TODO: Create name_scopes (for better graph visualisation)
        # TODO: Use buckets (better perfs)

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
                        localWt,  # Should have shape [num_classes, dim]
                        localB,
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
            if not self.args.test:  # TODO: Should use a placeholder instead
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

        # tensor的shape参数为一个张量，张量的阶可以理解为维数；
        # 0维[]表示纯量，即数；1维[i]表示向量，如[2]=>[3,9]；2维[i,j]表示矩阵，如[2,3]=>[[1,2,3],[4,5,6]]
        # None表示任意大小，空表示???表示0维即纯量，那[None,]和[None]是否一样呢？
        # 经实验证明[3,]与[3]完全一致，故推断[None,]与[None]含义相同，比较此处与tesorflow官方代码可知传入embedding_attention的参数类型
        # 几乎没有任何不同，说明之前关于数据维度过大导致显存内存占用剧增的推断应该不正确
        # 那么下一个不同的地方就是target的赋值，官方代码中target做了左偏移，后来又补充一位；
        # 还有一个不同是是否使用bucket，但是根据目前的了解使用只会提高运算速度，不应该可以减少内存占用？？？？这个问题很疑惑
        with tf.name_scope('placeholder_encoder'):
            self.encoderInputs = [tf.placeholder(tf.int32, [None, ]) for _ in
                                  range(self.args.maxLengthEnco)]  # Batch size * sequence length * input dim
            # [None,] batch_size,input dim
            # self.args.maxLengthEnco sequence length

        with tf.name_scope('placeholder_decoder'):
            self.decoderInputs = [tf.placeholder(tf.int32, [None, ], name='inputs') for _ in
                                  range(self.args.maxLengthDeco)]  # Same sentence length for input and output (Right ?)
            self.decoderTargets = [tf.placeholder(tf.int32, [None, ], name='targets') for _ in
                                   range(self.args.maxLengthDeco)]
            self.decoderWeights = [tf.placeholder(tf.float32, [None, ], name='weights') for _ in
                                   range(self.args.maxLengthDeco)]

        # Define the network
        # Here we use an embedding model, it takes integer as input and convert them into word vector for
        # better word representation

        # original system use embedding method only and can be improved by using attention, which could
        # improve the ability on longer sentences predictions.  *** embedding_attention_seq2seq ***

        # 注意如果要改成embedding_attention_seq2seq,其使用get_variable创建的变量name会不同（因为scope不同），直接运行
        # 会报错：ValueError: Variable embedding_rnn_seq2seq/rnn/embedding_wrapper/embedding does not exist,
        # or was not created with tf.get_variable(). Did you mean to set reuse=None in VarScope?
        # 即initEmbedding操作时reuse的变量不存在，应该加一个useAttention参数，并判断参数进行initEmbedding处的embedding
        # resign操作

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
            # Enable attentions mechanism, solve the problem of oom. Maybe could be solved by using buckets model.
            # Or maybe the problem is that inputs in a batch don't have the same length, that means we forgot the
            # padding operations, so that the dynamic model can't handle it. The bucket model can improve performance
            # by clustering the inputs which have similar length together. And it isn't necessary if we are not concern
            # about speed.
            # 在textdata里_createBatch中做了padding操作，问题就不在这里了。
            # 找到问题所在，attention必须配合Sampled softmax使用，可设置sample数目为256，可以明显看到计算速度比无attention机制时慢但是
            # 可以像tf官方model一样运行。
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

        # TODO: When the LSTM hidden size is too big, we should project the LSTM output into a smaller space (4086 => 2046): Should speed up
        # training and reduce memory usage. Other solution, use sampling softmax
        # For testing only
        if self.args.test:
            if not outputProjection:
                self.outputs = decoderOutputs
            else:
                self.outputs = [outputProjection(output) for output in decoderOutputs]

                # TODO: Attach a summary to visualize the output

        # For training only
        else:
            # Finally, we define the loss function
            # tf0.12 and before:self.lossFct = tf.nn.seq2seq.sequence_loss(decoderOutputs, self.decoderTargets,
            #                                                        self.decoderWeights,
            #                                                        self.textData.getVocabularySize())
            self.lossFct = tf.contrib.legacy_seq2seq.sequence_loss(
                decoderOutputs,
                self.decoderTargets,
                self.decoderWeights,
                self.textData.getVocabularySize(),
                softmax_loss_function=sampledSoftmax if outputProjection else None  # If None, use default SoftMax
            )
            tf.summary.scalar('loss', self.lossFct)  # Keep track of the cost
            # tf0.12 and before:tf.scalar_summary('loss', self.lossFct)  # Keep track of the cost

            # Initialize the optimizer
            opt = tf.train.AdamOptimizer(
                # 此处使用的优化算法与官方模型中不同，官方模型使用随机梯度下降方法GradientDescentOptimizer
                learning_rate=self.args.learningRate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-08
            )
            self.optOp = opt.minimize(self.lossFct)

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

            ops = (self.optOp, self.lossFct)
        else:  # Testing (batchSize == 1)

            for i in range(self.args.maxLengthEnco):
                feedDict[self.encoderInputs[i]] = batch.encoderSeqs[i]
            feedDict[self.decoderInputs[0]] = [self.textData.goToken]

            ops = (self.outputs,)

        # Return one pass operator
        return ops, feedDict
