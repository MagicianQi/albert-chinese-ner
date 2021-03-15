from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import os
import modeling
import optimization_finetuning as optimization
import tokenization
import tensorflow as tf
import pickle
import tf_metrics

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.
    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        # self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """Reads a BIO data."""
        with open(input_file, encoding='utf-8') as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                word = line.strip().split(' ')[0]
                label = line.strip().split(' ')[-1]
                if contends.startswith("-DOCSTART-"):
                    words.append('')
                    continue
                # if len(contends) == 0 and words[-1] == '。':
                if len(contends) == 0:
                    l = ' '.join([label for label in labels if len(label) > 0])
                    w = ' '.join([word for word in words if len(word) > 0])
                    lines.append([l, w])
                    words = []
                    labels = []
                    continue
                words.append(word)
                labels.append(label)
            return lines

    @classmethod
    def _read_data_str(cls, input_str_list):
        """Reads a BIO data."""
        lines = []
        words = []
        labels = []
        for line in input_str_list:
            contends = line.strip()
            word = line.strip()
            label = "O"
            if len(contends) == 0:
                l = ' '.join([label for label in labels if len(label) > 0])
                w = ' '.join([word for word in words if len(word) > 0])
                lines.append([l, w])
                words = []
                labels = []
                continue
            words.append(word)
            labels.append(label)
        return lines


class NerProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_test_examples(self, input_str):
        return self._create_example(
            self._read_data_str(list(input_str)), "test")

    def get_labels(self):
        return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples


def convert_single_example(ex_index, example, label_map, max_seq_length, tokenizer, mode):
    textlist = example.text.split(' ')
    labellist = example.label.split(' ')
    tokens = []
    labels = []
    # print(textlist)
    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        # print(token)
        tokens.extend(token)
        label_1 = labellist[i]
        # print(label_1)
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            # else:
            #     labels.append("X")
        # print(tokens, labels)
        # tokens = tokenizer.tokenize(example.text)
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    ntokens.append("[SEP]")
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    # label_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        # label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    # assert len(label_mask) == max_seq_length

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        # tf.logging.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        # label_mask = label_mask
    )
    return feature


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file, mode=None):
    """Convert a set of `InputExample`s to a TFRecord file."""
    label_map = {}
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    with open('albert_base_ner_checkpoints/label2id.pkl', 'wb') as w:
        pickle.dump(label_map, w)

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_map,
                                         max_seq_length, tokenizer, mode)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        # features["is_real_example"] = create_int_feature(
        #     [int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        # "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings, max_seq_length):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    output_layer = model.get_sequence_output()

    hidden_size = output_layer.shape[-1].value

    output_weight = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        output_layer = tf.reshape(output_layer, [-1, hidden_size])
        logits = tf.matmul(output_layer, output_weight, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        logits = tf.reshape(logits, [-1, max_seq_length, 11])

        log_probs = tf.nn.log_softmax(logits, axis=-1)
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_sum(per_example_loss)
        probabilities = tf.nn.softmax(logits, axis=-1)
        predict = tf.argmax(probabilities, axis=-1)
        return (loss, per_example_loss, logits, predict)


def layer_norm(input_tensor, name=None):
    """Run layer normalization on the last dimension of the tensor."""
    return tf.contrib.layers.layer_norm(
        inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, max_seq_length):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        # is_real_example = None
        # if "is_real_example" in features:
        #   is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        # else:
        #   is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, predicts) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings, max_seq_length=max_seq_length)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, logits):
                # def metric_fn(label_ids, logits):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                precision = tf_metrics.precision(label_ids, predictions, 11, [2, 3, 4, 5, 6, 7], average="macro")
                recall = tf_metrics.recall(label_ids, predictions, 11, [2, 3, 4, 5, 6, 7], average="macro")
                f = tf_metrics.f1(label_ids, predictions, 11, [2, 3, 4, 5, 6, 7], average="macro")
                #
                return {
                    "eval_precision": precision,
                    "eval_recall": recall,
                    "eval_f": f,
                    # "eval_loss": loss,
                }

            eval_metrics = (metric_fn, [per_example_loss, label_ids, logits])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions=predicts,
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_ids)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[num_examples, seq_length], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, mode):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    label_map = {}
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_map,
                                         max_seq_length, tokenizer, mode=mode)

        features.append(feature)
    return features


class NerModel:

    def __init__(self,
                 vocab_file="./albert_config/vocab.txt",
                 label2id_file='albert_base_ner_checkpoints/label2id.pkl',
                 bert_config_file="./albert_base_zh/albert_config_base.json",
                 max_seq_length=128,
                 output_dir="albert_base_ner_checkpoints",
                 init_checkpoint="albert_base_zh/albert_model.ckpt",
                 predict_batch_size=8,
                 iterations_per_loop=1000,
                 use_tpu=False):
        tokenization.validate_case_matches_checkpoint(True,
                                                      init_checkpoint)

        bert_config = modeling.BertConfig.from_json_file(bert_config_file)

        if max_seq_length > bert_config.max_position_embeddings:
            raise ValueError(
                "Cannot use sequence length %d because the BERT model "
                "was only trained up to sequence length %d" %
                (max_seq_length, bert_config.max_position_embeddings))

        self.max_seq_length = max_seq_length

        tf.gfile.MakeDirs(output_dir)
        self.processor = NerProcessor()
        self.label_list = self.processor.get_labels()
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

        tpu_cluster_resolver = None

        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        print("###tpu_cluster_resolver:", tpu_cluster_resolver)
        run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            master=None,
            model_dir=output_dir,
            save_checkpoints_steps=None,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=iterations_per_loop,
                num_shards=None,
                per_host_input_for_training=is_per_host))

        model_fn = model_fn_builder(
            bert_config=bert_config,
            num_labels=len(self.label_list) + 1,
            init_checkpoint=init_checkpoint,
            learning_rate=None,
            num_train_steps=None,
            num_warmup_steps=None,
            use_tpu=use_tpu,
            use_one_hot_embeddings=use_tpu,
            max_seq_length=max_seq_length)

        self.estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=None,
            eval_batch_size=None,
            predict_batch_size=predict_batch_size)

        with open(label2id_file, 'rb') as rf:
            label2id = pickle.load(rf)
            self.id2label = {value: key for key, value in label2id.items()}

    def predict(self, text):
        predict_examples = self.processor.get_test_examples(text)
        features = convert_examples_to_features(predict_examples, self.label_list,
                                                self.max_seq_length, self.tokenizer, mode="test")

        print("----------------------")
        print(type(features))
        print("======================")

        predict_input_fn = input_fn_builder(
            features=features,
            seq_length=self.max_seq_length,
            is_training=False,
            drop_remainder=False)

        result = self.estimator.predict(input_fn=predict_input_fn)
        predict_result = []

        for prediction in result:
            for id in prediction:
                if id != 0:
                    predict_result.append(self.id2label[id])

        predict_result.remove("[CLS]")
        predict_result.remove("[SEP]")
        text.replace(" ", "")

        return self.decode_ner_bio(text, predict_result)

    @staticmethod
    def decode_ner_bio(text, bio_label_list):
        res = {
            "PER": [],
            "LOC": [],
            "ORG": []
        }
        str_temp = ""
        str_type = ""
        for char, label in zip(list(text), bio_label_list):
            if "-" in label:
                pos, tp = label.strip().split("-")
                if pos == "B":
                    if len(str_temp) == 0:
                        str_type = tp
                        str_temp += char
                    else:
                        res[str_type].append(str_temp)
                        str_temp = ""
                        str_type = ""
                if pos == "I":
                    str_temp += char
            else:
                if len(str_temp) == 0:
                    pass
                else:
                    res[str_type].append(str_temp)
                    str_temp = ""
                    str_type = ""
        return res


if __name__ == "__main__":
    ner_model = NerModel(vocab_file="albert_base_zh/vocab.txt",
                         label2id_file='albert_base_ner_checkpoints/label2id.pkl',
                         bert_config_file="./albert_base_zh/albert_config_base.json",
                         max_seq_length=128,
                         output_dir="albert_base_ner_checkpoints",
                         init_checkpoint="albert_base_zh/albert_model.ckpt",
                         predict_batch_size=8,
                         iterations_per_loop=1000,
                         use_tpu=False)

    test_text = "昨天，十三届全国人大四次会议解放军和武警部队代表团新闻发言人吴谦接受媒体采访。 " \
                "他表示，我们愿以最大诚意、尽最大努力争取两岸和平统一的前景，但绝不容忍“台独”分裂势力分裂祖国。 "

    res = ner_model.predict(test_text)
    print(res)
