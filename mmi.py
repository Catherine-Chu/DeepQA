import tensorflow as tf
import heapq
import numpy as np
from chatbot import textdata
maxLengthEnco = 10
maxLengthDeco = maxLengthEnco+2
batchSize = 2
"""
encoderInputs:表示一组数据中每句输入的第i个单词组成的list
        A list of 1D int32 Tensors of shape [batch_size]
        list of 1维的tensor，每个tensor的shape维batch_size，list的总长度为seqWordsLength
        batchSize[sequence数量，每一个位置存一个int-wordid]*seqWordsLength[一个sequence中单词的数量]
        这里的tf=int32,以及None维是为任意batchSize留的，相当于tf.palceholder(tf.int32,[batchSize]) shape=[3]对应数据[1,2,3]
        根据之前的研究后面的'，'应该可有可无，也就是说对于每一个单词，创建了一个batchSize维的placeholder，batchSize维度可随意，且每个数据均为int（因为对应的只有一个wordid）
"""
encoderInputs = [tf.placeholder(tf.int32, [None, ]) for _ in range(maxLengthEnco)]
"""
decoderOutput:表示一组数据中根据每句输入的第i个单词预测出的相应每句回复第i个单词组成的list，且每一个单词的预测结果用概率list表示
        A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x num_decoder_symbols] containing the generated
        outputs.
        list of 2维的tensor，每个tensor的shape维batch_size*num_decoder_symbols，list的总长度为seqWordsLength
        batchSize[sequence数量]*numSymbols[所有可能的候选单词的概率]*seqWordsLength[一个sequence中单词的数量]
"""
#decoderOutputs = [tf.placeholder(tf.int32, [None, ], name='outputs') for _ in range(maxLengthDeco)]
decoderOutputs = [tf.placeholder(tf.float32, [None, None], name='outputs') for _ in range(maxLengthDeco)]
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# print(len(encoderInputs)) :10
# print(len(decoderOutput)) :12


"""
type:
[<tf.Tensor 'softmax_projection/add:0' shape=(?, 96895) dtype=float32>,
<tf.Tensor 'softmax_projection_1/add:0' shape=(?, 96895) dtype=float32>,
<tf.Tensor 'softmax_projection_2/add:0' shape=(?, 96895) dtype=float32>,
<tf.Tensor 'softmax_projection_3/add:0' shape=(?, 96895) dtype=float32>,
<tf.Tensor 'softmax_projection_4/add:0' shape=(?, 96895) dtype=float32>,
<tf.Tensor 'softmax_projection_5/add:0' shape=(?, 96895) dtype=float32>,
<tf.Tensor 'softmax_projection_6/add:0' shape=(?, 96895) dtype=float32>,
<tf.Tensor 'softmax_projection_7/add:0' shape=(?, 96895) dtype=float32>,
<tf.Tensor 'softmax_projection_8/add:0' shape=(?, 96895) dtype=float32>,
<tf.Tensor 'softmax_projection_9/add:0' shape=(?, 96895) dtype=float32>,
<tf.Tensor 'softmax_projection_10/add:0' shape=(?, 96895) dtype=float32>,
<tf.Tensor 'softmax_projection_11/add:0' shape=(?, 96895) dtype=float32>]
e.g.: ?=1
[array([[-5.82156897, -5.79414797,  4.73048544, ...,  0.15647288, 0.07721806,  0.0305429 ]], dtype=float32),
 array([[-5.87432051, -5.88782644,  8.10336971, ...,  0.01765282, 0.0120801 ,  0.03390714]], dtype=float32),
 array([[ -5.58487654e+00,  -5.61610031e+00,   8.62971115e+00, ..., -9.79384966e-03,   5.55315288e-04,   3.84603292e-02]], dtype=float32),
 array([[ -5.56967878e+00,  -5.60411882e+00,   8.97450829e+00, ..., -1.29399374e-02,  -4.42799786e-03,   3.87347974e-02]], dtype=float32),
 array([[-5.57684994, -5.61660862,  9.23896027, ..., -0.01542589, -0.00945735,  0.03922283]], dtype=float32),
 array([[-5.5720644 , -5.62147427,  9.5486908 , ..., -0.01906555, -0.01402829,  0.03943301]], dtype=float32),
 array([[-5.44984579, -5.51701021,  9.81782055, ..., -0.02578579,  -0.01779747,  0.03352182]], dtype=float32),
 array([[ -5.32753992,  -5.41743517,  10.24660206, ...,  -0.03939041, -0.02749607,   0.03013962]], dtype=float32),
 array([[ -5.44616508,  -5.55493212,  10.96794796, ...,  -0.04997132, -0.03515806,   0.03147997]], dtype=float32),
 array([[ -5.75935125,  -5.88682604,  11.9512043 , ...,  -0.05407051, -0.0385011 ,   0.02942809]], dtype=float32),
 array([[ -5.99670029,  -6.13637447,  12.63446999, ...,  -0.05560013, -0.0381282 ,   0.02981753]], dtype=float32),
 array([[ -6.07383156,  -6.21608162,  12.86668587, ...,  -0.05616176, -0.03717431,   0.03076505]], dtype=float32)]
e.g.: ?=2
list长度为12，表示12个单词;
每个单词对应一个list1，list1长度为2，表示并行数据数batch_size为2;
每个并行数据对应一个list2，list2长度为96895，表示词表大小的预测概率
list
[ list1-1： [l2-1:[len=96895],l2-2:[len=96895]],
  list1-2： [[ ],[ ]],
  ……
  list1-12：[[ ],[ ]]
]
"""


# 首先要处理掉decoderOutputs的batch
def splitOutBatches(decoderOuts):
    seqs=[]
    w=0
    for eachwords in decoderOuts:
        for eachone in range(batchSize):
            if w==0:
                seqs.append([])
            seqs[eachone].append(np.array(eachwords[eachone]))
        w += 1
    return seqs

# 还需要处理掉encoderInputs的batch，注意到他的结构也是转置过的
# 如果考虑双轮fore-back结构可能可以干脆在第二次fore-back之中将简单结构的input传进来，就不用那么麻烦处理了
# 注意到由于decoder2Nbest的输入必然是一次test中或一个step训练中的到的out，无论如何splitOutBatch都要做，
# 只是如果在第二轮fore-back的时候吧输出的简单形式传进来，那mmi计算会简单很多
def splitInBatches(encoderInts):
    #TODO: 待改动，encoderInts应为一个list<tensor>,tensor的shape为[batchSize]
    encoderSeqsT = []
    for i in range(batchSize):
        encoderSeqT = []
        for j in range(maxLengthEnco):
            encoderSeqT.append(encoderInts[j][i])
        encoderSeqsT.append(encoderSeqT)
    seqs = encoderSeqsT
    return seqs
def decoder2Nbest(decoderOuts, N, fore2back = False):
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
                    #a.clear()
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
        decoderOuts=splitOutBatches(decoderOuts)
        for decoderOutputs in decoderOuts:
            logitsseqs, sequences = step(decoderOutputs)
            Py_ifx.append(logitsseqs)
            batchSeqs.append(sequences)
    else:
        logitsseqs, sequences = step(decoderOuts)
        Py_ifx.extend(logitsseqs)
        batchSeqs.extend(sequences)
    return Py_ifx, batchSeqs


# def computeMI(encoInput, decoOutput, params = None):
#     sum_mi = 0.0
#     x_value_list = list(set(encoInput))
#     y_value_list = list(set(decoOutput))
#     Px = [len(encoInput[encoInput == xval]) / float(len(encoInput)) for xval in x_value_list]  # P(x)
#     Py = [len(decoOutput[decoOutput == yval]) / float(len(decoOutput)) for yval in y_value_list]  # P(y)
#     for i in range(len(x_value_list)):
#         if Px[i] == 0.:
#             continue
#         sy = decoOutput[encoInput == x_value_list[i]]
#         if len(sy) == 0:
#             continue
#         pxy = [len(sy[sy == yval]) / float(len(decoOutput)) for yval in y_value_list]  # p(x,y)
#         t = tf.divide(pxy[Py > 0.], tf.multiply(Py[Py > 0.], Px[i]))  # log(P(x,y)/( P(x)*P(y))
#         sum_mi += tf.reduce_sum(tf.multiply(pxy[t > 0], tf.log(t[t > 0])))  # sum ( P(x,y)* log(P(x,y)/( P(x)*P(y)) )
#     return sum_mi
def computeMI(py_ifx, px_ify, encoInput, decoOutput, params = None):
    if not params:
        params = [1/float(len(decoOutput)), 1/float(len(encoInput))]
    assert len(params) == 2
    return py_ifx+params[0]+px_ify*params[1]

feedDict = {}
encoderSeqs=[[1,1],[3,3],[2,3],[3,1],[0,2],[0,0],[0,0],[0,0],[0,0],[0,0]]
decoderSeqs=[np.array([[0.01,0.92,0.02,0.05],[0.02,0.90,0.03,0.05]]),
             np.array([[0.02,0.01,0.12,0.87],[0.00,0.07,0.12,0.81]]),
             np.array([[0.23,0.01,0.75,0.01],[0.00,0.07,0.12,0.81]]),
             np.array([[0.00,0.07,0.12,0.81],[0.01,0.92,0.02,0.05]]),
             np.array([[0.99,0.00,0.00,0.01],[0.23,0.01,0.75,0.01]]),
             np.array([[0.99,0.00,0.00,0.01],[0.99,0.00,0.00,0.01]]),
             np.array([[0.99,0.00,0.00,0.01],[0.99,0.00,0.00,0.01]]),
             np.array([[0.99,0.00,0.00,0.01],[0.99,0.00,0.00,0.01]]),
             np.array([[0.99,0.00,0.00,0.01],[0.99,0.00,0.00,0.01]]),
             np.array([[0.99,0.00,0.00,0.01],[0.99,0.00,0.00,0.01]]),
             np.array([[0.99,0.00,0.00,0.01],[0.99,0.00,0.00,0.01]]),
             np.array([[0.99,0.00,0.00,0.01],[0.99,0.00,0.00,0.01]])
             ]
for i in range(maxLengthEnco):
    feedDict[encoderInputs[i]] = encoderSeqs[i]
for j in range(maxLengthDeco):
    feedDict[decoderOutputs[j]] = decoderSeqs[j]
outs = sess.run(decoderOutputs, feedDict)
py_ifx, batchseqs = decoder2Nbest(outs, 5)
inOneBatchs = splitInBatches(encoderInputs)
for onebatch in range(len(py_ifx)):
    inco=textdata.sequence2str(inOneBatchs[onebatch], clean=True)
    MI=[]
    for each in range(len(batchseqs[onebatch])):
        deco=textdata.sequence2str(batchseqs[onebatch][each], clean=True)
        pyi_ifx=py_ifx[onebatch][each]
        px_ifyi=0 #TODO: x=[pi,qi],y=a,实现waston mode，训练出模型seq2seq(qi|a),计算px_ifyi for all candidates yi
        MI.append(computeMI(pyi_ifx,px_ifyi,inco,deco))
# def main():
#     encoderInputs=feedict
#     decoderOutputs=model out
#     nbestout=decoder2Nbest(decoderOutputs) # splitOutBatchs被调用
#     if fore2back:
#         if not onlyone:
#             # 即每次输入一个batch的输入，和相应输出的batch * n-best candidates（这个也可以直接不输入了，因为上面的函数返回值本来就是这个）
#             easy_encoderInputs=feedict1
#             i=0
#             for conv in nbestout:
#                 for response in conv:
#                     computeMI(easy_encoderInputs[i],response)
#                 i +=1
#         else:
#             # 即每次输入一个输入，和相应输出的n-best candidates
#             easy_encoderInputs = feedict1
#             easy_decoderOutputs = feedict2
#             for response in easy_decoderOutputs:
#                 computeMI(easy_encoderInputs, response)
#     else:
#         easy_encoderInputs=splitInBatches(encoderInputs)
#         i = 0
#         for conv in nbestout:
#             for response in conv:
#                 computeMI(easy_encoderInputs[i], response)
#             i += 1