import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, DistilBertModel, AlbertModel, DistilBertPreTrainedModel
import torch.nn.functional as F

PRETRAINED_MODEL_MAP = {
    'bert': BertModel,
    'distilbert': DistilBertModel,
    'albert': AlbertModel
}

class JointBERT(BertPreTrainedModel):
    def __init__(self, bert_config, args, intent_label_lst, all_docs=None):#config配置
    # def __init__(self, bert_config, args, slot_label_lst):  # config配置
        super(JointBERT, self).__init__(bert_config)#继承父类属性bert_config
        self.args = args
        self.num_intent_labels = len(intent_label_lst)#intenet_label_lst获取所有意图标签
        self.num_filters = 256

        self.filter_sizes = [2] #   不同窗口大小
        if self.args.cat == 'cat':
            self.hidden_size = 768 * 2
        elif  self.args.cat == 'add':
            self.hidden_size = 768

        if args.do_pred:
            self.bert = PRETRAINED_MODEL_MAP[args.model_type](config=bert_config)#BertModel(config=bert_config)
        else:
            self.bert = PRETRAINED_MODEL_MAP[args.model_type].from_pretrained(args.model_name_or_path, config=bert_config)  # Load pretrained bert

    def forward(self, input_ids_1, attention_mask_1, token_type_ids_1, input_ids_2, attention_mask_2, token_type_ids_2, intent_label_ids):
        outputs_1 = self.bert(input_ids_1, attention_mask=attention_mask_1, token_type_ids=token_type_ids_1)  # sequence_output, pooled_output, (hidden_states), (attentions)
        outputs_2 = self.bert(input_ids_2, attention_mask=attention_mask_2, token_type_ids=token_type_ids_2)  # sequence_output, pooled_output, (hidden_states), (attentions)

        w1_output = outputs_1[1]  # dim=2
        w2_output = outputs_2[1]  # dim=2

        if self.args.cat == 'cat':
            outputs = torch.cat((w1_output, w2_output), dim=-1)
        elif self.args.cat == 'add':
            outputs = torch.add(w1_output, w2_output)
        else:
            print("The cat methor should be 'cat' or 'add'")
            breakpoint()

        return outputs


class JointDistilBERT(DistilBertPreTrainedModel):
    def __init__(self, distilbert_config, args, intent_label_lst, slot_label_lst):
        super(JointDistilBERT, self).__init__(distilbert_config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        if args.do_pred:
            self.distilbert = PRETRAINED_MODEL_MAP[args.model_type](config=distilbert_config)
        else:
            self.distilbert = PRETRAINED_MODEL_MAP[args.model_type].from_pretrained(args.model_name_or_path,
                                                                                    config=distilbert_config)  # Load pretrained bert


        self.slot_pad_token_idx = slot_label_lst.index(args.slot_pad_label)

    def forward(self, input_ids, attention_mask, intent_label_ids, slot_labels_ids):
        outputs = self.distilbert(input_ids, attention_mask=attention_mask)  # last-layer hidden-state, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0]  # [CLS]

        outputs = None
        return outputs







