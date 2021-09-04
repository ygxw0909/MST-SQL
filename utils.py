import os
import json
import unicodedata
import re
import transformers
import numpy as np
import copy
import random

pretrained_weights = {
    ("bert", "base"): "bert-base-uncased",
    ("bert", "large"): "bert-large-uncased-whole-word-masking",
    ("roberta", "base"): "roberta-base",
    ("roberta", "large"): "roberta-large",
    ("albert", "xlarge"): "albert-xlarge-v2",
    ("roberta_cn", "large"): "/home/gxn/ST-SQL-Final/pretrained/roberta_cn/",
    ("albert", "xlarge"): "albert-xlarge-v2",
    ("grappa", "large"): "Salesforce/grappa_large_jnt",
    ("tapas", "base"): "google/tapas-base",
}

def read_jsonl(jsonl):
    for line in open(jsonl, encoding="utf8"):
        sample = json.loads(line.rstrip())
        yield sample

def read_conf(conf_path):
    config = {}
    for line in open(conf_path, encoding="utf8"):
        if line.strip() == "" or line[0] == "#":
             continue
        fields = line.strip().split()
        config[fields[0]] = fields[1]
    config["train_data_path"] = os.path.abspath(config["train_data_path"])
    config["dev_data_path"] = os.path.abspath(config["dev_data_path"])
    config["test_data_path"] = os.path.abspath(config["test_data_path"])

    return config

def create_base_model(config):
    weights_name = pretrained_weights[(config["base_class"], config["base_name"])]
    if config["base_class"] == "bert":
        return transformers.BertModel.from_pretrained(weights_name)
    elif config["base_class"] == "roberta":
        return transformers.RobertaModel.from_pretrained(weights_name)
    elif config["base_class"] == "roberta_cn":
        config = transformers.BertConfig.from_pretrained(weights_name + "config.json")
        return transformers.BertModel.from_pretrained(weights_name + "pytorch_model.bin", config=config)
    elif config["base_class"] == "albert":
        return transformers.AlbertModel.from_pretrained(weights_name)
    elif config["base_class"] == "grappa":
        return transformers.AutoModel.from_pretrained(weights_name)
    elif config["base_class"] == "tapas":
        return transformers.TapasModel.from_pretrained(weights_name)
    else:
        raise Exception("base_class {0} not supported".format(config["base_class"]))

def create_tokenizer(config):
    weights_name = pretrained_weights[(config["base_class"], config["base_name"])]
    if config["base_class"] == "bert":
        return transformers.BertTokenizer.from_pretrained(weights_name)
    elif config["base_class"] == "roberta":
        return transformers.RobertaTokenizer.from_pretrained(weights_name)
    elif config["base_class"] == "roberta_cn":
        return transformers.BertTokenizer.from_pretrained(weights_name)
    elif config["base_class"] == "albert":
        return transformers.AlbertTokenizer.from_pretrained(weights_name)
    elif config["base_class"] == "grappa":
        return transformers.RobertaTokenizer.from_pretrained(weights_name)
    elif config["base_class"] == "tapas":
        return transformers.TapasTokenizer.from_pretrained(weights_name)
    else:
        raise Exception("base_class {0} not supported".format(config["base_class"]))

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\n" or c == "\r":
        return True
    cat = unicodedata.category(c)
    if cat == "Zs":
        return True
    return False

def is_punctuation(c):
  """Checks whether `chars` is a punctuation character."""
  cp = ord(c)
  # We treat all non-letter/number ASCII as punctuation.
  # Characters such as "^", "$", and "`" are not in the Unicode
  # Punctuation class but we treat them as punctuation anyways, for
  # consistency.
  if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
      (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
    return True
  cat = unicodedata.category(c)
  if cat.startswith("P") or cat.startswith("S"):
    return True
  return False

def basic_tokenize(doc):
    doc_tokens = []
    char_to_word = []
    word_to_char_start = []
    prev_is_whitespace = True
    prev_is_punc = False
    prev_is_num = False
    for pos, c in enumerate(doc):
        if is_whitespace(c):
            prev_is_whitespace = True
            prev_is_punc = False
        else:
            if prev_is_whitespace or is_punctuation(c) or prev_is_punc or (prev_is_num and not str(c).isnumeric()):
                doc_tokens.append(c)
                word_to_char_start.append(pos)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
            prev_is_punc = is_punctuation(c)
            prev_is_num = str(c).isnumeric()
        char_to_word.append(len(doc_tokens) - 1)

    return doc_tokens, char_to_word, word_to_char_start

def basic_tokenize_cn(doc):
    doc_tokens, char_to_word, word_to_char_start = [], [], []
    for pos, c in enumerate(doc):
        doc_tokens.append(c)
        char_to_word.append(pos)
        word_to_char_start.append(pos)
    return doc_tokens, char_to_word, word_to_char_start

def edit_distance(s1, s2):
    len1 = len(s1)
    len2 = len(s2)
    dp = np.zeros((len1 + 1, len2 + 1))
    for i in range(len1 + 1):
        dp[i][0] = i
    for i in range(len2 + 1):
        dp[0][i] = i
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if s1[i - 1] == s2[j - 1]:
                temp = 0
            else:
                temp = 1
            dp[i][j] = min(dp[i - 1][j - 1] + temp,
                           min(dp[i - 1][j] + 1, dp[i][j - 1] + 1))
    return dp[len1][len2]


def longest_common_subsequence(s1, s2):
    len1 = len(s1)
    len2 = len(s2)
    dp = np.zeros((len1 + 1, len2 + 1))
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[len1][len2]


def literal_exact_match(q_str, c_str):
    if q_str.find(c_str) != -1:
        return True
    return False


def literal_score_match(q_tok_wp, c_tok_wp):
    q_len = len(q_tok_wp)
    c_str = " ".join(c_tok_wp)
    c_str_len = len(c_str)
    max_score = -1
    st, ed = -1, -1
    for n in range(len(c_tok_wp), 0, -1):
        for i in range(q_len):
            if i + n > q_len:
                break
            q_str = " ".join(q_tok_wp[i: i + n])
            q_str_len = len(q_str)
            lcs = longest_common_subsequence(q_str, c_str)
            assert q_str_len > 0 and c_str_len > 0

            score = (lcs * 1.0 / q_str_len + lcs * 1.0 / c_str_len) / 2.0
            if score > max_score:
                max_score = score
                st = i
                ed = i + n
                if max_score == 1.0:
                    return max_score, st, ed
    return max_score, st, ed

def filter_content_one_column(tokenizer, q_tok_cn, cells, threshold, max_num):
    q_str_cn = "".join(q_tok_cn).lower()
    q_tok_wp = []
    for tok in q_tok_cn:
        sub_toks = tokenizer.tokenize(tok.lower())
        for sub_tok in sub_toks:
            q_tok_wp.append(sub_tok)
    matching = []
    for cell in cells:
        content = str(cell).lower()
        if q_str_cn.find(re.compile(' ').sub('', content)) == -1:
            matching.append([str(cell), 0.0])
            continue
        c_tok_wp = tokenizer.tokenize(content)
        max_score, _, _ = literal_score_match(q_tok_wp, c_tok_wp)
        matching.append([str(cell), max_score])
    # print("matching: ", matching)
    matching = sorted(matching, key=lambda x:x[1], reverse=True)
    # print(matching)
    res = []
    for i, elem in enumerate(matching):
        if i >= max_num:
            break
        res.append(elem[0])
    return res