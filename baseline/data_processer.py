import sys
import collections
import numpy
import random
import math
import os
import gc
from evaluator import *

def read_input_files(file_paths, max_sentence_length=-1):
    """
    Reads input files in whitespace-separated format.
    Will split file_paths on comma, reading from multiple files.
    The format assumes the first column is the word, the last column is the label.
    """
    sentences = []
    line_length = None
    for file_path in file_paths.strip().split(","):
        with open(file_path, "r") as f:
            sentence = []
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    line_parts = line.split()
                    assert(len(line_parts) >= 2)
                    assert(len(line_parts) == line_length or line_length == None)
                    line_length = len(line_parts)
                    sentence.append(line_parts)
                elif len(line) == 0 and len(sentence) > 0:
                    if max_sentence_length <= 0 or len(sentence) <= max_sentence_length:
                        sentences.append(sentence)
                    sentence = []
            if len(sentence) > 0:
                if max_sentence_length <= 0 or len(sentence) <= max_sentence_length:
                    sentences.append(sentence)
    return sentences


def process_sentences(data, labeler, is_training, learningrate, config, name):
    """
    #处理句子数据成为batch
    """
    evaluator = SequenceLabelingEvaluator("1", labeler.label2id, False)
    batches_of_sentence_ids = create_batches_of_sentence_ids(data, 3)
    #print(batches_of_sentence_ids)
    if is_training == True:
        random.shuffle(batches_of_sentence_ids)

    for sentence_ids_in_batch in batches_of_sentence_ids:
        batch = [data[i] for i in sentence_ids_in_batch]
        #print(batch)
        #print(batch)
        cost, predicted_labels, predicted_probs = labeler.process_batch(batch, is_training, learningrate)

        evaluator.append_data(cost, batch, predicted_labels)

        word_ids, char_ids, char_mask, label_ids = None, None, None, None

    results = evaluator.get_results(name)
    for key in results:
        print(key + ": " + str(results[key]))

    return results


def create_batches_of_sentence_ids(sentences, max_batch_size):
    """
    #产生一个batch的句子id的列表
    """
    batch_of_sentence_ids = []
    current_batch = []
    max_sequence_length = 0
    for i in range(len(sentences)):
        current_batch.append(i)
        if len(sentences[i]) >= max_sequence_length:
            max_sequence_length = len(sentences[i])
        if len(current_batch) >= max_batch_size:
            # print(batch_of_sentence_ids)
            # print(current_batch)
            batch_of_sentence_ids.append(current_batch)
            current_batch = []
            max_sequence_length = 0

    #剩余不足一个batch的数据加入到batch中

    if len(current_batch) > 0 :
        batch_of_sentence_ids.append(current_batch)
    return batch_of_sentence_ids
