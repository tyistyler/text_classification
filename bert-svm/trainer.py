import os
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup

from utils import MODEL_CLASSES, set_seed, compute_metrics, get_intent_labels

from sklearn import svm
from sklearn.decomposition import PCA
import joblib

logger = logging.getLogger(__name__)

#Trainer(args, train_dataset, dev_dataset, test_dataset)
class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.intent_label_lst = get_intent_labels(args)#获取所有意图标签
        # self.slot_label_lst = get_slot_labels(args)#获取所有槽值标签
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = args.ignore_index# -100

        self.best_precision = 0
        self.best_epoch = 0

        self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]#BertConfig, JointBERT， BertTokenizer
        self.bert_config = self.config_class.from_pretrained(args.model_name_or_path, finetuning_task=args.task)
        #here
        self.model = self.model_class(self.bert_config, args, self.intent_label_lst)#JointBERT(BertConfig, args, intent_label_lst, slot_label_lst)
        self.svm_model = svm.SVC(C=1.0, kernel=self.args.kernel)
        self.pca = PCA(n_components=200)
        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

    def train(self):
        # 定义采样方式，对象为样本特征
        train_sampler = RandomSampler(self.train_dataset)
        # 构建dataloader，dataloader本质是一个可迭代对象,batch_size=16
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps# 10000
            # 10000/500/2=10
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:# 500/2*10
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)优化器
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        tr_loss = 0.0
        train_x = None
        train_y = None
        self.model.zero_grad()#梯度归0

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")#进度条<--tqdm, 例如分成10组

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")#进度条, desc为进度条前缀
            for step, batch in enumerate(epoch_iterator):
                self.model.eval()#训练模式
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU  把输入数据 放到设备上
                # 分类任务是单句子，不需要设置token_type_ids 默认全为0，batch[2]
                # 定义模型的输入参数 字典形式
                with torch.no_grad():  # 关闭梯度计算
                    inputs = {'input_ids_1': batch[0],# 第一行
                              'attention_mask_1': batch[1],# 第二行
                              'token_type_ids_1': batch[2],
                              'input_ids_2': batch[3],  # 第一行
                              'attention_mask_2': batch[4],  # 第二行
                              'token_type_ids_2': batch[5],
                              'intent_label_ids': batch[6]
                              }

                    outputs = self.model(**inputs)#得到模型输出JointBert(input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids, corpus_label_ids)
                # outputs-->[16, 768]
                    if train_x is None:
                        train_x = outputs.detach().cpu().numpy()
                        train_y = batch[6].detach().cpu().numpy()
                    else:
                        train_x = np.append(train_x, outputs.detach().cpu().numpy(), axis=0)
                        train_y = np.append(train_y, batch[6].detach().cpu().numpy(), axis=0)

                global_step += 1
                print("global_step:",global_step)
                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            print(train_x.shape)
            print(train_y.shape)
            train_x = self.pca.fit_transform(train_x)   # 降维
            print(train_x.shape)
            # self.svm_model.fit(train_x, train_y)
            print("训练一轮结束，开始测试")
            # precision = self.svm_model.score(train_x, train_y)
            # print("BERT-SVM的准确率为:",precision)
            # joblib.dump(self.svm_model, 'ckpt/svm.pkl')

            precision = self.evaluate("dev")  # fine-tuning
            if precision >= self.best_precision:
                self.best_precision = precision
                self.best_epoch = _
                joblib.dump(self.svm_model, args.model_dir+'/svm.pkl')
                
                print('save model finished')
            print('best p_score is', self.best_precision)
            print('best epoch is', self.best_epoch)


            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

        return global_step

    def evaluate(self, mode):#test
        # We use test dataset because semeval doesn't have dev dataset
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.batch_size)

        dev_x = None
        dev_y = None

        self.model.eval()#验证模式

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():# 关闭梯度计算
                inputs = {'input_ids_1': batch[0],
                          'attention_mask_1': batch[1],
                          'token_type_ids_1': batch[2],
                          'input_ids_2': batch[3],  # 第一行
                          'attention_mask_2': batch[4],  # 第二行
                          'token_type_ids_2': batch[5],
                          'intent_label_ids': batch[6]
                          }
                outputs = self.model(**inputs)
                if dev_x is None:
                    dev_x = outputs.detach().cpu().numpy()
                    dev_y = batch[6].detach().cpu().numpy()
                else:
                    dev_x = np.append(dev_x, outputs.detach().cpu().numpy(), axis=0)
                    dev_y = np.append(dev_y, batch[6].detach().cpu().numpy(), axis=0)
        print("dev_x:",dev_x.shape)
        dev_x = self.pca.fit_transform(dev_x)   # 降维
        print("dev_x:",dev_x.shape)
        precision = self.svm_model.score(dev_x, dev_y)

        logger.info("***** Eval results *****")

        if mode == 'test':
            f = open('result/result.txt', 'a', encoding='utf-8')
            f.write("precision:     ",precision)
            f.write("\n")
            f.close()
        return precision

#     def save_model(self):
#         # Save model checkpoint (Overwrite)
#         output_dir = os.path.join(self.args.model_dir)

#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
#         model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
#         model_to_save.save_pretrained(output_dir)
#         torch.save(self.args, os.path.join(output_dir, 'training_config.bin'))
#         logger.info("Saving model checkpoint to %s", output_dir)

#     def load_model(self):
#         # Check whether model exists
#         if not os.path.exists(self.args.model_dir):
#             raise Exception("Model doesn't exists! Train first!")

#         try:
#             self.bert_config = self.config_class.from_pretrained(self.args.model_dir)
#             logger.info("***** Config loaded *****")
#             self.model = self.model_class.from_pretrained(self.args.model_dir, config=self.bert_config,
#                                     args=self.args, intent_label_lst=self.intent_label_lst
#                                     )
#             # self.model = self.model_class.from_pretrained(self.args.model_dir, config=self.bert_config,
#             #                                               args=self.args,slot_label_lst=self.slot_label_lst)
#             self.model.to(self.device)
#             logger.info("***** Model Loaded *****")
#         except:
#             raise Exception("Some model files might be missing...")

#     def _convert_texts_to_tensors(self, text, tokenizer,
#                                   cls_token_segment_id=0,
#                                   pad_token_segment_id=0,
#                                   sequence_a_segment_id=0,
#                                   mask_padding_with_zero=True):
#         """
#         Only add input_ids, attention_mask, token_type_ids
#         Labels aren't required.
#         """
#         # Setting based on the current model type
#         cls_token = tokenizer.cls_token
#         sep_token = tokenizer.sep_token
#         unk_token = tokenizer.unk_token
#         pad_token_id = tokenizer.pad_token_id

#         input_ids_1_batch = []
#         attention_mask_1_batch = []
#         token_type_ids_1_batch = []

#         input_ids_2_batch = []
#         attention_mask_2_batch = []
#         token_type_ids_2_batch = []

#         slot_label_mask_batch = []

#         tokens_1 = []
#         tokens_2 = []
#             # slot_label_mask = []
#         words_1, words_2 = text.split()
#         for w1, w2 in zip(words_1, words_2):

#             word_tokens_1 = tokenizer.tokenize(w1)
#             word_tokens_2 = tokenizer.tokenize(w2)
#             if not word_tokens_1:
#                 word_tokens_1 = [unk_token]  # For handling the bad-encoded word
#             if not word_tokens_2:
#                 word_tokens_2 = [unk_token]
#             tokens_1.extend(word_tokens_1)
#             tokens_2.extend(word_tokens_2)
#                 # Real label position as 0 for the first token of the word, and padding ids for the remaining tokens
#                 # slot_label_mask.extend([0] + [self.pad_token_label_id] * (len(word_tokens) - 1))

#             # Account for [CLS] and [SEP]
#         special_tokens_count = 2
#         if len(tokens_1) > self.args.max_seq_len - special_tokens_count:
#             tokens_1 = tokens_1[:(self.args.max_seq_len - special_tokens_count)]
#         if len(tokens_2) > self.args.max_seq_len - special_tokens_count:
#             tokens_2 = tokens_2[:(self.args.max_seq_len - special_tokens_count)]


#             # Add [SEP] token
#         tokens_1 += [sep_token]
#         tokens_2 += [sep_token]
#             # slot_label_mask += [self.pad_token_label_id]
#         token_type_ids_1 = [sequence_a_segment_id] * len(tokens_1)
#         token_type_ids_2 = [sequence_a_segment_id] * len(tokens_2)
#         # Add [CLS] token
#         tokens_1 = [cls_token] + tokens_1
#         token_type_ids_1 = [cls_token_segment_id] + token_type_ids_1

#         tokens_2 = [cls_token] + tokens_1
#         token_type_ids_2 = [cls_token_segment_id] + token_type_ids_2

#         input_ids_1 = tokenizer.convert_tokens_to_ids(tokens_1)
#         input_ids_2 = tokenizer.convert_tokens_to_ids(tokens_2)
#         # The mask has 1 for real tokens and 0 for padding tokens. Only real
#         # tokens are attended to.非填充部分的token对应1
#         attention_mask_1 = [1 if mask_padding_with_zero else 0] * len(input_ids_1)
#         attention_mask_2 = [1 if mask_padding_with_zero else 0] * len(input_ids_2)

#         # Zero-pad up to the sequence length.
#         padding_length_1 = self.args.max_seq_len - len(input_ids_1)
#         input_ids_1 = input_ids_1 + ([pad_token_id] * padding_length_1)
#         attention_mask_1 = attention_mask_1 + ([0 if mask_padding_with_zero else 1] * padding_length_1)
#         token_type_ids_1 = token_type_ids_1 + ([pad_token_segment_id] * padding_length_1)

#         padding_length_2 = self.args.max_seq_len - len(input_ids_2)
#         input_ids_2 = input_ids_2 + ([pad_token_id] * padding_length_2)
#         attention_mask_2 = attention_mask_2 + ([0 if mask_padding_with_zero else 1] * padding_length_2)
#         token_type_ids_2 = token_type_ids_2 + ([pad_token_segment_id] * padding_length_2)
#         # slot_label_mask = slot_label_mask + ([self.pad_token_label_id] * padding_length)

#         input_ids_1_batch.append(input_ids_1)
#         attention_mask_1_batch.append(attention_mask_1)
#         token_type_ids_1_batch.append(token_type_ids_1)

#         input_ids_2_batch.append(input_ids_2)
#         attention_mask_2_batch.append(attention_mask_2)
#         token_type_ids_2_batch.append(token_type_ids_2)
            
#             # slot_label_mask_batch.append(slot_label_mask)

#         # Making tensor that is batch size of 1
#         input_ids_1_batch = torch.tensor(input_ids_1_batch, dtype=torch.long).to(self.device)
#         attention_mask_1_batch = torch.tensor(attention_mask_1_batch, dtype=torch.long).to(self.device)
#         token_type_ids_1_batch = torch.tensor(token_type_ids_1_batch, dtype=torch.long).to(self.device)

#         input_ids_2_batch = torch.tensor(input_ids_2_batch, dtype=torch.long).to(self.device)
#         attention_mask_2_batch = torch.tensor(attention_mask_2_batch, dtype=torch.long).to(self.device)
#         token_type_ids_2_batch = torch.tensor(token_type_ids_2_batch, dtype=torch.long).to(self.device)

#         # slot_label_mask_batch = torch.tensor(slot_label_mask_batch, dtype=torch.long).to(self.device)

#         # dataset = TensorDataset(input_ids_batch, attention_mask_batch, token_type_ids_batch)

#         return input_ids_1_batch, attention_mask_1_batch, token_type_ids_1_batch, input_ids_2_batch, attention_mask_2_batch, token_type_ids_2_batch

#     def predict(self, texts, tokenizer):
#         batch = self._convert_texts_to_tensors(texts, tokenizer)
#         # print(batch[0])
#         # print(batch[1])
#         # print(batch[2])
#         # Predict
#         # sampler = SequentialSampler(dataset)
#         # data_loader = DataLoader(dataset, sampler=sampler, batch_size=self.args.batch_size)

#         intent_preds = None
#         # slot_label_mask = batch[3]
#         self.model.eval()

#         # for batch in tqdm(data_loader, desc='Predicting'):
#             # batch = tuple(t.to(self.device) for t in batch)
#         with torch.no_grad():
#             inputs = {'input_ids_1': batch[0],
#                       'attention_mask_1': batch[1],
#                       'token_type_ids_1':batch[2],
#                       'input_ids_2': batch[3],
#                       'attention_mask_2': batch[4],
#                       'token_type_ids_2': batch[5],
#                       'intent_label_ids': None
#                       }

#             outputs = self.model(**inputs)
#             _, (intent_logits, slot_logits) = outputs[:2]  # loss doesn't needed

#         # Intent prediction
#         intent_preds = intent_logits.detach().cpu().numpy()
#         intent_preds = np.argmax(intent_preds, axis=1)
#         intent_list = []
#         for intent_idx in intent_preds:
#             intent_list.append(self.intent_label_lst[intent_idx])

#         # print(intent_list)
#         return intent_list[0]



