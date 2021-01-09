# export CUDA_VISIBLE_DEVICES=1

import os
seed = 66
model_dir = "model-1"

#       s01
#       cat
os.system("python3 main.py --task s01 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.2 "
          "--cat cat --cnn_dropout 0.3 --do_train --do_eval --use_cnn".format(seed))

os.system("python3 main.py --task s01 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.2 "
          "--cat cat --cnn_dropout 0.4 --do_train --do_eval --use_cnn".format(seed))

os.system("python3 main.py --task s01 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.3 "
          "--cat cat --cnn_dropout 0.3 --do_train --do_eval --use_cnn".format(seed))

os.system("python3 main.py --task s01 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.3 "
          "--cat cat --cnn_dropout 0.4 --do_train --do_eval --use_cnn".format(seed))

#       add
os.system("python3 main.py --task s01 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.2 "
          "--cat add --cnn_dropout 0.3 --do_train --do_eval --use_cnn".format(seed))

os.system("python3 main.py --task s01 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.2 "
          "--cat add --cnn_dropout 0.4 --do_train --do_eval --use_cnn".format(seed))

os.system("python3 main.py --task s01 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.3 "
          "--cat add --cnn_dropout 0.3 --do_train --do_eval --use_cnn".format(seed))

os.system("python3 main.py --task s01 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.3 "
          "--cat add --cnn_dropout 0.4 --do_train --do_eval --use_cnn".format(seed))


#       s02
#       cat
os.system("python3 main.py --task s02 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.2 "
          "--cat cat --cnn_dropout 0.3 --do_train --do_eval --use_cnn".format(seed))

os.system("python3 main.py --task s02 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.2 "
          "--cat cat --cnn_dropout 0.4 --do_train --do_eval --use_cnn".format(seed))

os.system("python3 main.py --task s02 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.3 "
          "--cat cat --cnn_dropout 0.3 --do_train --do_eval --use_cnn".format(seed))

os.system("python3 main.py --task s02 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.3 "
          "--cat cat --cnn_dropout 0.4 --do_train --do_eval --use_cnn".format(seed))

#       add
os.system("python3 main.py --task s02 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.2 "
          "--cat add --cnn_dropout 0.3 --do_train --do_eval --use_cnn".format(seed))

os.system("python3 main.py --task s02 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.2 "
          "--cat add --cnn_dropout 0.4 --do_train --do_eval --use_cnn".format(seed))

os.system("python3 main.py --task s02 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.3 "
          "--cat add --cnn_dropout 0.3 --do_train --do_eval --use_cnn".format(seed))

os.system("python3 main.py --task s02 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.3 "
          "--cat add --cnn_dropout 0.4 --do_train --do_eval --use_cnn".format(seed))


#       s03
#       cat
os.system("python3 main.py --task s03 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.2 "
          "--cat cat --cnn_dropout 0.3 --do_train --do_eval --use_cnn".format(seed))

os.system("python3 main.py --task s03 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.2 "
          "--cat cat --cnn_dropout 0.4 --do_train --do_eval --use_cnn".format(seed))

os.system("python3 main.py --task s03 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.3 "
          "--cat cat --cnn_dropout 0.3 --do_train --do_eval --use_cnn".format(seed))

os.system("python3 main.py --task s03 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.3 "
          "--cat cat --cnn_dropout 0.4 --do_train --do_eval --use_cnn".format(seed))

#       add
os.system("python3 main.py --task s03 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.2 "
          "--cat add --cnn_dropout 0.3 --do_train --do_eval --use_cnn".format(seed))

os.system("python3 main.py --task s03 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.2 "
          "--cat add --cnn_dropout 0.4 --do_train --do_eval --use_cnn".format(seed))

os.system("python3 main.py --task s03 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.3 "
          "--cat add --cnn_dropout 0.3 --do_train --do_eval --use_cnn".format(seed))

os.system("python3 main.py --task s03 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.3 "
          "--cat add --cnn_dropout 0.4 --do_train --do_eval --use_cnn".format(seed))

#       s04
#       cat
os.system("python3 main.py --task s04 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.2 "
          "--cat cat --cnn_dropout 0.3 --do_train --do_eval --use_cnn".format(seed))

os.system("python3 main.py --task s04 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.2 "
          "--cat cat --cnn_dropout 0.4 --do_train --do_eval --use_cnn".format(seed))

os.system("python3 main.py --task s04 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.3 "
          "--cat cat --cnn_dropout 0.3 --do_train --do_eval --use_cnn".format(seed))

os.system("python3 main.py --task s04 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.3 "
          "--cat cat --cnn_dropout 0.4 --do_train --do_eval --use_cnn".format(seed))

#       add
os.system("python3 main.py --task s04 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.2 "
          "--cat add --cnn_dropout 0.3 --do_train --do_eval --use_cnn".format(seed))

os.system("python3 main.py --task s04 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.2 "
          "--cat add --cnn_dropout 0.4 --do_train --do_eval --use_cnn".format(seed))

os.system("python3 main.py --task s04 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.3 "
          "--cat add --cnn_dropout 0.3 --do_train --do_eval --use_cnn".format(seed))

os.system("python3 main.py --task s04 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.3 "
          "--cat add --cnn_dropout 0.4 --do_train --do_eval --use_cnn".format(seed))
          
#       s05
#       cat
os.system("python3 main.py --task s05 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.2 "
          "--cat cat --cnn_dropout 0.3 --do_train --do_eval --use_cnn".format(seed))

os.system("python3 main.py --task s05 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.2 "
          "--cat cat --cnn_dropout 0.4 --do_train --do_eval --use_cnn".format(seed))

os.system("python3 main.py --task s05 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.3 "
          "--cat cat --cnn_dropout 0.3 --do_train --do_eval --use_cnn".format(seed))

os.system("python3 main.py --task s05 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.3 "
          "--cat cat --cnn_dropout 0.4 --do_train --do_eval --use_cnn".format(seed))

#       add
os.system("python3 main.py --task s05 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.2 "
          "--cat add --cnn_dropout 0.3 --do_train --do_eval --use_cnn".format(seed))

os.system("python3 main.py --task s05 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.2 "
          "--cat add --cnn_dropout 0.4 --do_train --do_eval --use_cnn".format(seed))

os.system("python3 main.py --task s05 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.3 "
          "--cat add --cnn_dropout 0.3 --do_train --do_eval --use_cnn".format(seed))

os.system("python3 main.py --task s05 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.3 "
          "--cat add --cnn_dropout 0.4 --do_train --do_eval --use_cnn".format(seed))

#       s06
#       cat
os.system("python3 main.py --task s06 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.2 "
          "--cat cat --cnn_dropout 0.3 --do_train --do_eval --use_cnn".format(seed))

os.system("python3 main.py --task s06 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.2 "
          "--cat cat --cnn_dropout 0.4 --do_train --do_eval --use_cnn".format(seed))

os.system("python3 main.py --task s06 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.3 "
          "--cat cat --cnn_dropout 0.3 --do_train --do_eval --use_cnn".format(seed))

os.system("python3 main.py --task s06 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.3 "
          "--cat cat --cnn_dropout 0.4 --do_train --do_eval --use_cnn".format(seed))

#       add
os.system("python3 main.py --task s06 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.2 "
          "--cat add --cnn_dropout 0.3 --do_train --do_eval --use_cnn".format(seed))

os.system("python3 main.py --task s06 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.2 "
          "--cat add --cnn_dropout 0.4 --do_train --do_eval --use_cnn".format(seed))

os.system("python3 main.py --task s06 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.3 "
          "--cat add --cnn_dropout 0.3 --do_train --do_eval --use_cnn".format(seed))

os.system("python3 main.py --task s06 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.3 "
          "--cat add --cnn_dropout 0.4 --do_train --do_eval --use_cnn".format(seed))

#       s07
#       cat
os.system("python3 main.py --task s07 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.2 "
          "--cat cat --cnn_dropout 0.3 --do_train --do_eval --use_cnn".format(seed))

os.system("python3 main.py --task s07 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.2 "
          "--cat cat --cnn_dropout 0.4 --do_train --do_eval --use_cnn".format(seed))

os.system("python3 main.py --task s07 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.3 "
          "--cat cat --cnn_dropout 0.3 --do_train --do_eval --use_cnn".format(seed))

os.system("python3 main.py --task s07 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.3 "
          "--cat cat --cnn_dropout 0.4 --do_train --do_eval --use_cnn".format(seed))

#       add
os.system("python3 main.py --task s07 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.2 "
          "--cat add --cnn_dropout 0.3 --do_train --do_eval --use_cnn".format(seed))

os.system("python3 main.py --task s07 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.2 "
          "--cat add --cnn_dropout 0.4 --do_train --do_eval --use_cnn".format(seed))

os.system("python3 main.py --task s07 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.3 "
          "--cat add --cnn_dropout 0.3 --do_train --do_eval --use_cnn".format(seed))

os.system("python3 main.py --task s07 --model_type bert "
          "--seed {} --batch_size 64 --learning_rate 5e-5 --cls_dropout 0.3 "
          "--cat add --cnn_dropout 0.4 --do_train --do_eval --use_cnn".format(seed))



# python main.py --task sememe --model_type bert --model_dir patent_model --do_pred
