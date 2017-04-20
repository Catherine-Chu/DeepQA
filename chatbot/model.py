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
                        localB,  # [dim]
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

                if not outputProjection:
                    self.outputs = decoderOutputs
                else:
                    self.outputs = [outputProjection(output) for output in decoderOutputs]

                # self.mmiReward的计算与赋值不在Model中，仅在训练调用agentStep之后，通过返回的action（即output）计算
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

    def agentStep(self, batch, terminate=False):

        feedDict = {}
        ops = None

        if not self.args.test and (self.args.train == 'matualinfo' or self.args.train == 'reinforcement'):  # Training
            if not terminate:
                for i in range(self.args.maxLengthEnco):
                    feedDict[self.encoderInputs[i]] = batch.encoderSeqs[i]
                feedDict[self.decoderInputs[0]] = [self.textData.goToken]
                for i in range(self.args.maxLengthDeco):
                    feedDict[self.decoderWeights[i]] = batch.weights[i]
                ops = (self.outputs,)  # action
            else:
                ops = (self.mmiOp,)  # last action and optimizer object
        else:  # Testing (batchSize == 1)
            for i in range(self.args.maxLengthEnco):
                feedDict[self.encoderInputs[i]] = batch.encoderSeqs[i]
            feedDict[self.decoderInputs[0]] = [self.textData.goToken]

            ops = (self.outputs,)

        return ops, feedDict
