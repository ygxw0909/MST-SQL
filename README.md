# MST-SQL
Data and Code for paper *Improving Few-Shot Text-to-SQL with Meta Self-Training via Column Specificity* is available for research purposes.

# Requirement
* python 3.7.11
* pytorch 1.8.1
* transformers 4.5.0
* torch_scatter 2.0.7
* records
* babel
* tabulate

# Data
In our experiments, we used WikiSQL, ESQL as benchmarks, and WikiTableQuestions as semi-supervised data. All of which can be downloaded by [here](https://drive.google.com/drive/folders/1nnN2Yph_FGxisPo4_SwkWQ4YO8E69-AX?usp=sharing). After obtaining the data, make a new file named "data" in root directory, and put the data into that file.

# Train
#### 1. WikiSQL
Execute the following command for training ST-SQL on WikiSQL.
```bash
python main.py --conf conf/wikisql.conf
```

#### 2. ESQL
Put the Chinese version of RoBERTa in the appropriate directory and modify the settings in util.py. Execute the following command for training ST-SQL on ESQL.
```bash
python main.py --conf conf/esql.conf
```

The pre-training model can be changed following the weight_name in util.py. If you want to train the model on few-shot or zero-shot setting, the data path and some parameters mentioned in our paper need to be changed as well.

# Test
If you want to test the model with Execution Guiding(EG), and use Logical Form accuracy(LF) and Execution accuracy(EX), use following command after modifying some parameters in the code.
```bash
python wikisql_prediction.py
```

If you want to test fast without EG and EX, run following command.
```bash
python wikisql_prediction_simple.py
```

