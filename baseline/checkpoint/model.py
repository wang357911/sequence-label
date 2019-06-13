import collections
import tensorflow as tf
import re
import numpy
import os

try:
    import cPickle as pickle
except:
    import pickle

#WORD_EMBEDDING = "./glove"

class Basicmodel(object):
    
    def __init__(self, config):
        self.config = config
        self.UNK = "<unk>"
        self.pad = "<pad>"
        self.label_pad = "pad"
        # self.sos = "<sos>"
        # self.eos = "<eos>"
        self.word2id = None
        self.label2id = None
    
    def build_vocabs(self, data_train, embedding_path=None):
        """
        处理单词和标签，整合成字典，embedding矩阵
        
        """
        #产生词汇字典
        data_source = list(data_train)
        word_counter = collections.Counter()
        for sentence in data_source:
            for word in sentence:
                w = word[0].lower()         #单词做小写转化
                w = re.sub(r'\d', '0', w)   #去除数字
                word_counter[w] += 1
        self.word2id = collections.OrderedDict([(self.UNK, 0),(self.pad, 1)])
        for word, count in word_counter.most_common():
            if count >= 0:              #数据集中出现频率
                if word not in self.word2id:
                    self.word2id[word] = len(self.word2id)
        #print(self.word2id)
        
        self.singletons = set([word for word in word_counter if word_counter[word] == 1]) #只出现一次的词
        
        #产生标签字典
        label_counter = collections.Counter()
        for sentence in data_train: #this one only based on training data
            for word in sentence:
                label_counter[word[-1]] += 1
        self.label2id = collections.OrderedDict([(self.label_pad, 0)])
        for label, count in label_counter.most_common():
            if label not in self.label2id:
                self.label2id[label] = len(self.label2id)

        #创建预训练embedding的vc表
        if embedding_path != None:
            self.embedding_vocab = set([self.UNK])
            with open(embedding_path, "r") as f:
                for line in f:
                    line_parts = line.strip().split()
                    if len(line_parts) <=2:
                        continue
                    w = line_parts[0].lower()
                    w = re.sub(r'\d', '0', w)
                    self.embedding_vocab.add(w)
            #print(self.embedding_vocab)
            word2id_revised = collections.OrderedDict()
            for word in self.word2id:
                if word in self.embedding_vocab and word not in word2id_revised:
                    word2id_revised[word] = len(word2id_revised)
            #print(word2id_revised)
            self.word2id = word2id_revised
        if not os.path.exists("./vocab/trans.pkl"):
            with open("./vocab/trans.pkl", "wb") as f:
                pickle.dump((self.word2id, self.label2id), f)

        print("n_words: " + str(len(self.word2id)))
        print("n_labels: " + str(len(self.label2id)))

    def load(self):
        with open("./vocab/trans.pkl", "rb") as r:
            self.word2id, self.label2id = pickle.load(r)

    def trans2id(self, data):
        sentences = []
        labels = []
        for sen in data:
            sen_ids = []
            label_ids = []
            for token in sen:
                if token[0] in self.word2id:
                    sen_ids.append(self.word2id[token[0]])
                else:
                    sen_ids.append(1)
                label_ids.append(self.label2id[token[-1]])
            sentences.append(sen_ids)
            labels.append(label_ids)
        sentences = tf.keras.preprocessing.sequence.pad_sequences(sentences, maxlen=self.config["max_sentence_length"], \
                                                                  padding='post', truncating='post', dtype="int32",
                                                                  value=1)
        labels = tf.keras.preprocessing.sequence.pad_sequences(labels, maxlen=self.config["max_sentence_length"], \
                                                               padding='post', truncating='post', dtype="int32",
                                                               value=0)

        return sentences, labels


    def build_data(self, data):
        sentences = []
        labels = []
        for sen in data:
            sen_ids = []
            label_ids = []
            for token in sen:
                if token[0] in self.word2id:
                    sen_ids.append(self.word2id[token[0]])
                else:
                    sen_ids.append(1)
                label_ids.append(self.label2id[token[-1]])
            sentences.append(sen_ids)
            labels.append(label_ids)
        sentences = tf.keras.preprocessing.sequence.pad_sequences(sentences, maxlen=self.config["max_sentence_length"],\
                                                                  padding='post', truncating='post', dtype="int32", value=1)
        labels = tf.keras.preprocessing.sequence.pad_sequences(labels, maxlen=self.config["max_sentence_length"], \
                                                                  padding='post', truncating='post',dtype="int32", value=0)

        print(len(sentences))
        train_dataset = tf.data.Dataset.from_tensor_slices((sentences, labels))
        train_dataset = train_dataset.batch(64)
        train_dataset = train_dataset.repeat()
        train_dataset = train_dataset.prefetch(64)
        self.iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        self.train_initializer = self.iterator.make_initializer(train_dataset)

    
    def construct_network(self):
        self.word_ids = tf.placeholder(tf.int32, [None, 10], name="word_ids")
        self.label_ids = tf.placeholder(tf.int32, [None, 10], name="word_ids")
        self.keep_prob = tf.placeholder(tf.float64, [], name="keep_prob")
        self.loss = 0.0
        self.global_step = tf.Variable(-1, trainable=False, name='global_step')

        #构建embedding矩阵
        self.initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1)  #定义初始化方式
        if self.config["use_pretrain_embedding"]:
            embedding_vector = numpy.ones(shape= [len(self.word2id), self.config["word_embedding_size"]])
            with open("vocab_text.txt") as f:
                vocabs = f.readlines()
                for vocab in vocabs:
                    index = self.word2id[vocab.split()[0]]
                    embedding_vector[index] = numpy.array(vocab.split()[-1].split(","), dtype=numpy.float64)
            self.word_embeddings = tf.get_variable(name= "embedding", initializer=embedding_vector, dtype=tf.float64, \
                                                   trainable=True)
        else:
            self.word_embeddings = tf.get_variable("word_embeddings",
                shape=[len(self.word2id), self.config["word_embedding_size"]],
                initializer=self.initializer, trainable=True )

        #定义网络
        # with tf.variable_scope('inputs'):
        #     x, y_label = self.iterator.get_next()
        self.input_tensor = tf.nn.embedding_lookup(self.word_embeddings, self.word_ids)

        #定义biLSTM
        fw_cell = tf.nn.rnn_cell.LSTMCell(self.config["word_recurrent_size"], reuse=tf.AUTO_REUSE)
        fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=self.keep_prob)
        bw_cell = tf.nn.rnn_cell.LSTMCell(self.config["word_recurrent_size"], reuse=tf.AUTO_REUSE)
        bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=self.keep_prob)

        self.output, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, self.input_tensor, dtype = tf.float64)

        self.output = tf.stack(self.output, axis=1)

        self.output = tf.reshape(self.output, [-1, self.config["word_recurrent_size"] * 2])

        with tf.variable_scope('outputs'):
            w = tf.get_variable(shape = (self.config["word_recurrent_size"] * 2,len(self.label2id)), \
                                initializer= tf.random_normal_initializer(), \
                                dtype=tf.float64, name = "w")
            b = tf.get_variable(shape= len(self.label2id), dtype = tf.float64, initializer=tf.zeros_initializer(), name = "b")
            y = tf.matmul(self.output, w) + b
            self.y_predict = tf.cast(tf.argmax(y, axis=1), tf.int32)

        self.y_label_reshape = tf.cast(tf.reshape(self.label_ids, [-1]), tf.int32)
        self.correct_prediction = tf.equal(self.y_predict, self.y_label_reshape)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float64))

        self.cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_label_reshape,
                                                                                      logits=tf.cast(y, tf.float64)))
        tf.summary.scalar('loss', self.cross_entropy)
        tf.summary.histogram('y_predict', self.y_predict)
        tf.summary.scalar('accuracy', self.accuracy)
        self.summaries = tf.summary.merge_all()
        self.train = tf.train.AdamOptimizer(self.config["learning_rate"]).minimize(self.cross_entropy, \
                                                                                   global_step=self.global_step)


    def initialize_session(self):
        tf.set_random_seed(100)
        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True
        session_config.gpu_options.per_process_gpu_memory_fraction = 1
        self.session = tf.Session(config=session_config)
        self.session.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter(os.path.join(self.config["summary_dir"], 'train'),
                                            self.session.graph)
        self.saver = tf.train.Saver(max_to_keep=1)
        
    def get_parameter_count(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            print(variable)
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        return total_parameters


    def translate2id(self, sens):
        """
        将句子列表转化成id列表
        :param sens:
        :return: 返回id列表
        """

        id_lists = []
        for sen in sens:
            id_list = []
            token_list = sen.split()
            for token in token_list:
                token = token.lower()
                token = re.sub(r'\d', '0', token)
                if token in self.word2id:
                    id_list.append(self.word2id[token])
                else:
                    id_list.append(self.word2id[self.UNK])
            id_lists.append(id_list)
        #print(id_lists)
        id_lists = tf.keras.preprocessing.sequence.pad_sequences(id_lists, maxlen=self.config["max_sentence_length"], \
                                                      padding='post', truncating='post', dtype="int32", value=1)
        return id_lists

    def id2label(self, ids_list):
        """
        将输出的id转化成原始标签
        :param ids_list:
        :return:
        """

        label_lists = []
        for ids in ids_list:
            label_list = []
            for id in ids:
                if not id == 0:
                    label = list(self.label2id.keys())[list(self.label2id.values()).index(id)]
                    label_list.append(label)
            label_lists.append(label_list)

        return label_lists

