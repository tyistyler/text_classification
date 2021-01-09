# export CUDA_VISIBLE_DEVICES=1

import os
seed = 66
model_dir = "model-1"

#       sememe
#       cat
os.system("python3 main.py --task sememe --model_type bert "
          "--seed {} --batch_size 1000 --learning_rate 5e-5 "
          "--cat cat --do_train --use_svm --kernel linear".format(seed))

os.system("python3 main.py --task sememe --model_type bert "
          "--seed {} --batch_size 1000 --learning_rate 5e-5 "
          "--cat cat --do_train --use_svm --kernel poly".format(seed))

os.system("python3 main.py --task sememe --model_type bert "
          "--seed {} --batch_size 1000 --learning_rate 5e-5 "
          "--cat cat --do_train --use_svm --kernel rbf".format(seed))

os.system("python3 main.py --task sememe --model_type bert "
          "--seed {} --batch_size 1000 --learning_rate 5e-5 "
          "--cat cat --do_train --use_svm --kernel sigmoid".format(seed))

os.system("python3 main.py --task sememe --model_type bert "
          "--seed {} --batch_size 1000 --learning_rate 5e-5 "
          "--cat cat --do_train --use_svm --kernel precomputed".format(seed))

#       sememe
#       add
os.system("python3 main.py --task sememe --model_type bert "
          "--seed {} --batch_size 1000 --learning_rate 5e-5 "
          "--cat add --do_train --use_svm --kernel linear".format(seed))

os.system("python3 main.py --task sememe --model_type bert "
          "--seed {} --batch_size 1000 --learning_rate 5e-5 "
          "--cat add --do_train --use_svm --kernel poly".format(seed))

os.system("python3 main.py --task sememe --model_type bert "
          "--seed {} --batch_size 1000 --learning_rate 5e-5 "
          "--cat add --do_train --use_svm --kernel rbf".format(seed))

os.system("python3 main.py --task sememe --model_type bert "
          "--seed {} --batch_size 1000 --learning_rate 5e-5 "
          "--cat add --do_train --use_svm --kernel sigmoid".format(seed))

os.system("python3 main.py --task sememe --model_type bert "
          "--seed {} --batch_size 1000 --learning_rate 5e-5 "
          "--cat add --do_train --use_svm --kernel precomputed".format(seed))
          
#       s01
#       cat
os.system("python3 main.py --task s01 --model_type bert "
          "--seed {} --batch_size 1000 --learning_rate 5e-5 "
          "--cat cat --do_train --use_svm --kernel linear".format(seed))

os.system("python3 main.py --task s01 --model_type bert "
          "--seed {} --batch_size 1000 --learning_rate 5e-5 "
          "--cat cat --do_train --use_svm --kernel poly".format(seed))

os.system("python3 main.py --task s01 --model_type bert "
          "--seed {} --batch_size 1000 --learning_rate 5e-5 "
          "--cat cat --do_train --use_svm --kernel rbf".format(seed))

os.system("python3 main.py --task s01 --model_type bert "
          "--seed {} --batch_size 1000 --learning_rate 5e-5 "
          "--cat cat --do_train --use_svm --kernel sigmoid".format(seed))

os.system("python3 main.py --task s01 --model_type bert "
          "--seed {} --batch_size 1000 --learning_rate 5e-5 "
          "--cat cat --do_train --use_svm --kernel precomputed".format(seed))

#       s01
#       add
os.system("python3 main.py --task s01 --model_type bert "
          "--seed {} --batch_size 1000 --learning_rate 5e-5 "
          "--cat add --do_train --use_svm --kernel linear".format(seed))

os.system("python3 main.py --task s01 --model_type bert "
          "--seed {} --batch_size 1000 --learning_rate 5e-5 "
          "--cat add --do_train --use_svm --kernel poly".format(seed))

os.system("python3 main.py --task s01 --model_type bert "
          "--seed {} --batch_size 1000 --learning_rate 5e-5 "
          "--cat add --do_train --use_svm --kernel rbf".format(seed))

os.system("python3 main.py --task s01 --model_type bert "
          "--seed {} --batch_size 1000 --learning_rate 5e-5 "
          "--cat add --do_train --use_svm --kernel sigmoid".format(seed))

os.system("python3 main.py --task s01 --model_type bert "
          "--seed {} --batch_size 1000 --learning_rate 5e-5 "
          "--cat add --do_train --use_svm --kernel precomputed".format(seed))         
          
          
          