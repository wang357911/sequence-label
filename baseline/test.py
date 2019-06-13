from model import *
from data_processer import *
import tensorflow as tf
from tqdm import tqdm

import numpy


#print(tf.keras.__version__)
EMBEDDING = "./glove.6B.300d.txt"


config = {}
config["use_pretrain_embedding"] = True
config["vocab_size"] = 30000
config["word_embedding_size"] = 40
config["word_recurrent_size"] = 40
config["dropout_input"] = 0.5
config["dropout_word_lstm"] = 0.5
config["max_sentence_length"] = 10
config["summary_dir"] = "./tensorboard_log"
config["learning_rate"] = 0.001
config["ckpt_dir"] = "./checkpoint"
is_training = True


data_train = read_input_files("./train.in")
test_data = data_train[:3]


st_model = Basicmodel(config)
if os.path.exists("./vocab"):
    st_model.load()
else:
    st_model.build_vocabs(data_train)
st_model.build_data(data_train)
st_model.construct_network()
st_model.initialize_session()


print(st_model.word2id)
print(st_model.label2id)

# print(st_model.translate2id(["a,b,c,d"]))
# print("-----")
# print(st_model.id2label([[4,3,2,1,0,0,0,0,0,0]]))


gstep = 0

print(st_model.get_parameter_count())

if is_training:
    if tf.gfile.Exists(config["summary_dir"]):
        tf.gfile.DeleteRecursively(config["summary_dir"])
    if not tf.gfile.Exists(config["ckpt_dir"]):
        os.mkdir(config["ckpt_dir"])
    else:
        tf.gfile.DeleteRecursively(config["ckpt_dir"])

    for epoch in tqdm(range(50000)):
        tf.train.global_step(st_model.session, global_step_tensor=st_model.global_step)
        st_model.session.run(st_model.train_initializer)
        for step in range(20):
            x, y_label = st_model.session.run(st_model.iterator.get_next())
            smrs, loss, acc, gstep, _ = st_model.session.run(\
                [st_model.summaries, st_model.cross_entropy, st_model.accuracy, st_model.global_step, st_model.train],\
                                             feed_dict={st_model.keep_prob: 0.8, \
                                                        st_model.word_ids: x,\
                                                        st_model.label_ids: y_label})
            st_model.writer.add_summary(smrs, gstep)

        # st_model.session.run(st_model.accuracy, feed_dict={st_model.keep_prob: 1,
        #                                                    st_model.word_ids: x, \
        #                                                    st_model.label_ids: y_label\
        #                                                    })
        if epoch % 30 ==0:
            result = st_model.session.run(st_model.y_predict, feed_dict={st_model.keep_prob: 1,
                                    st_model.word_ids: st_model.translate2id(["a b c d a"]), \
                                    st_model.label_ids: numpy.array([[2, 4, 3, 1, 0, 0, 0, 0, 0, 0]], dtype=numpy.uint32) \
                                                                })
            print(st_model.id2label([result]))
            st_model.saver.save(st_model.session, os.path.join(config["ckpt_dir"], "ckpt"), global_step=gstep)

else:
    ckpt = tf.train.get_checkpoint_state(config["ckpt_dir"])
    st_model.saver.restore(st_model.session, ckpt.model_checkpoint_path)
    result = st_model.session.run(st_model.y_predict, feed_dict={st_model.keep_prob: 1,
                                                                 st_model.word_ids: st_model.translate2id(["a b c d a"]), \
                                                                 st_model.label_ids: numpy.array([[2, 4, 3, 1, 0, 0, 0, 0, 0, 0]],
                                                                     dtype=numpy.uint32) \
                                                                 })
    print(st_model.id2label([result]))

