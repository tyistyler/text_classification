import os
import copy
import json
import logging

import torch
from torch.utils.data import TensorDataset

from utils import get_intent_labels

logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        intent_label: (Optional) string. The intent label of the example.
        slot_labels: (Optional) list. The slot labels of the example.
    """

    def __init__(self, guid, words, intent_label=None, slot_labels=None):
    # def __init__(self, guid, words, slot_labels=None):
        self.guid = guid
        self.words = words
        self.intent_label = intent_label
        self.slot_labels = slot_labels

    def __repr__(self):#此时终端会打印出信息
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)#深拷贝，创建了一个新的字典
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string.将此实例序列化为JSON字符串"""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"#indent是缩进打印


class InputFeatures(object):
    """A single set of features of data.一组特征数据"""

    def __init__(self, input_ids_1, attention_mask_1, token_type_ids_1, input_ids_2, attention_mask_2, token_type_ids_2, intent_label_id):
    # def __init__(self, input_ids, attention_mask, token_type_ids, slot_labels_ids):
        self.input_ids_1 = input_ids_1
        self.attention_mask_1 = attention_mask_1
        self.token_type_ids_1 = token_type_ids_1

        self.input_ids_2= input_ids_2
        self.attention_mask_2 = attention_mask_2
        self.token_type_ids_2 = token_type_ids_2

        self.intent_label_id = intent_label_id
        # self.slot_labels_ids = slot_labels_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class JointProcessor(object):
    """Processor for the JointBERT data set.处理器 """

    def __init__(self, args):
        self.args = args
        self.intent_labels = get_intent_labels(args)#获得文档中的意图标签
        # self.slot_labels = get_slot_labels(args)#获得文档中的槽标签

        self.input_text_file = 'seq.in'#输入句子
        self.intent_label_file = 'label.txt'#句子标签
        self.slot_labels_file = 'seq.out'#句子槽值

    @classmethod#不需要实例化
    def _read_file(cls, input_file, quotechar=None):#读取文件
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_examples(self, texts, intents, set_type):
    # def _create_examples(self, texts, slots, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, (text, intent) in enumerate(zip(texts,intents)):
            guid = "%s-%s" % (set_type, i)
            # 1. input_text
            words = text.split()  # ['被录取', '榜上有名']
            # 2. intent
            intent_label = self.intent_labels.index(intent) if intent in self.intent_labels else self.intent_labels.index("UNK")
            # 3. slot
            # slot_labels = []
            # for s in slot.split():
            #     slot_labels.append(self.slot_labels.index(s) if s in self.slot_labels else self.slot_labels.index("UNK"))

            # assert len(words) == len(slot_labels)#不满足条件，触发异常；相等时说明每个词都有对应的槽值
            examples.append(InputExample(guid=guid, words=words, intent_label=intent_label))
            # examples.append(InputExample(guid=guid, words=words, slot_labels=slot_labels))
        return examples

    #获得训练内容，如\data\atis\train\label
    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        data_path = os.path.join(self.args.data_dir, self.args.task, mode)
        logger.info("LOOKING AT {}".format(data_path))
        return self._create_examples(texts=self._read_file(os.path.join(data_path, self.input_text_file)),
                                     intents=self._read_file(os.path.join(data_path, self.intent_label_file)),
                                     set_type=mode)


processors = {
    "sememe": JointProcessor,
    "s01": JointProcessor,
    "s02": JointProcessor,
    "s03": JointProcessor,
    "s04": JointProcessor,
    "s05": JointProcessor,
    "s06": JointProcessor,
    "s07": JointProcessor
}

# 把InputExamples对象,转换为输入特征InputFeatures
def convert_examples_to_features(examples, max_seq_len, tokenizer,
                                 pad_token_label_id=-100,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):
    # Setting based on the current model type.根据当前模型类型进行设置
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # Tokenize word by word (for NER)
        tokens_1 = []
        tokens_2 = []
        words_1 = example.words[0]
        words_2 = example.words[1]


        for w1, w2 in zip(words_1, words_2):

            word_tokens_1 = tokenizer.tokenize(w1)
            word_tokens_2 = tokenizer.tokenize(w2)
            if not word_tokens_1:
                word_tokens_1 = [unk_token]  # For handling the bad-encoded word
            if not word_tokens_2:
                word_tokens_2 = [unk_token]
            tokens_1.extend(word_tokens_1)
            tokens_2.extend(word_tokens_2)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            # slot_labels_ids.extend([int(slot_label)] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        # 单词太多，截取前一部分
        if len(tokens_1) > max_seq_len - special_tokens_count:
            tokens_1 = tokens_1[:(max_seq_len - special_tokens_count)]
        if len(tokens_2) > max_seq_len - special_tokens_count:
            tokens_2 = tokens_2[:(max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens_1 += [sep_token]
        tokens_2 += [sep_token]
        # slot_labels_ids += [pad_token_label_id]
        # 对于句子对任务，属于句子A的token为0，句子B的token为1；对于分类任务，只有一个输入句子，全为0
        token_type_ids_1 = [sequence_a_segment_id] * len(tokens_1)
        token_type_ids_2 = [sequence_a_segment_id] * len(tokens_2)
        # Add [CLS] token
        tokens_1 = [cls_token] + tokens_1
        token_type_ids_1 = [cls_token_segment_id] + token_type_ids_1

        tokens_2 = [cls_token] + tokens_2
        token_type_ids_2 = [cls_token_segment_id] + token_type_ids_2

        # token在词汇表中的索引
        input_ids_1 = tokenizer.convert_tokens_to_ids(tokens_1)
        input_ids_2 = tokenizer.convert_tokens_to_ids(tokens_2)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.非填充部分的token对应1
        attention_mask_1 = [1 if mask_padding_with_zero else 0] * len(input_ids_1)
        attention_mask_2 = [1 if mask_padding_with_zero else 0] * len(input_ids_2)

        # Zero-pad up to the sequence length.
        padding_length_1 = max_seq_len - len(input_ids_1)
        input_ids_1 = input_ids_1 + ([pad_token_id] * padding_length_1)
        attention_mask_1 = attention_mask_1 + ([0 if mask_padding_with_zero else 1] * padding_length_1)
        token_type_ids_1 = token_type_ids_1 + ([pad_token_segment_id] * padding_length_1)

        padding_length_2 = max_seq_len - len(input_ids_2)
        input_ids_2 = input_ids_2 + ([pad_token_id] * padding_length_2)
        attention_mask_2 = attention_mask_2 + ([0 if mask_padding_with_zero else 1] * padding_length_2)
        token_type_ids_2 = token_type_ids_2 + ([pad_token_segment_id] * padding_length_2)


        # 确保长度相同
        assert len(input_ids_1) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids_1), max_seq_len)
        assert len(attention_mask_1) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask_1), max_seq_len)
        assert len(token_type_ids_1) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids_1), max_seq_len)
        # assert len(slot_labels_ids) == max_seq_len, "Error with slot labels length {} vs {}".format(len(slot_labels_ids), max_seq_len)

        intent_label_id = int(example.intent_label)


        # 前5个样本,打印处理效果
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens_1]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids_1]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask_1]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids_1]))
            logger.info("intent_label: %s (id = %d)" % (example.intent_label, intent_label_id))
            # logger.info("slot_labels: %s" % " ".join([str(x) for x in slot_labels_ids]))



        features.append(
            InputFeatures(input_ids_1=input_ids_1,
                          attention_mask_1=attention_mask_1,
                          token_type_ids_1=token_type_ids_1,
                          input_ids_2=input_ids_2,
                          attention_mask_2=attention_mask_2,
                          token_type_ids_2=token_type_ids_2,
                          intent_label_id=intent_label_id
                          # slot_labels_ids=slot_labels_ids,
                          ))
    return features


def load_and_cache_examples(args, tokenizer, mode):#mode=train/dev/test
    processor = processors[args.task](args)#获取自定义任务处理器，我们要处理atis数据集JointProcessor(args)

    # Load data features from cache or dataset file，特征保存目录的命名
    cached_features_file = os.path.join(
        args.data_dir,
        'cached_{}_{}_{}_{}'.format(
            mode,
            args.task,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.max_seq_len
        )
    )
    # 如果对数据集已经构造好特征了,直接加载,避免重复处理
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    # 否则对数据集进行处理,得到特征
    else:
        # Load data features from dataset file
        logger.info("Creating features from dataset file at %s", args.data_dir)
        # 对验证集、测试集、训练集进行处理,把数据转换为InputExample对象
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, dev, test is available")

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        pad_token_label_id = args.ignore_index
        features = convert_examples_to_features(examples, args.max_seq_len, tokenizer,
                                                pad_token_label_id=pad_token_label_id)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)


    # Convert to Tensors and build dataset
    all_input_ids_1 = torch.tensor([f.input_ids_1 for f in features], dtype=torch.long)
    all_attention_mask_1 = torch.tensor([f.attention_mask_1 for f in features], dtype=torch.long)
    all_token_type_ids_1 = torch.tensor([f.token_type_ids_1 for f in features], dtype=torch.long)

    all_input_ids_2 = torch.tensor([f.input_ids_2 for f in features], dtype=torch.long)
    all_attention_mask_2 = torch.tensor([f.attention_mask_2 for f in features], dtype=torch.long)
    all_token_type_ids_2 = torch.tensor([f.token_type_ids_2 for f in features], dtype=torch.long)

    all_intent_label_ids = torch.tensor([f.intent_label_id for f in features], dtype=torch.long)
    # all_slot_labels_ids = torch.tensor([f.slot_labels_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids_1, all_attention_mask_1,all_token_type_ids_1,
                            all_input_ids_2, all_attention_mask_2,all_token_type_ids_2, all_intent_label_ids)
    return dataset


