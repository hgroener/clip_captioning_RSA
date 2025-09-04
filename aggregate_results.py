import pandas as pd

## aggregating scores
score_list = [{"sampling": "greedy", "RSA": "no", "t": "n.a.", "POS": "n.a." , "CIDEr": 0, "informativity": 0},
              {"sampling": "greedy", "RSA": "yes", "t": "no", "POS": "no" , "CIDEr": 0, "informativity": 0},
              {"sampling": "greedy", "RSA": "yes", "t": 0.5, "POS": "no" , "CIDEr": 0, "informativity": 0},
              {"sampling": "greedy", "RSA": "yes", "t": "no", "POS": "yes" , "CIDEr": 0, "informativity": 0},
              {"sampling": "greedy", "RSA": "yes", "t": "yes", "POS": "yes" , "CIDEr": 0, "informativity": 0},
              {"sampling": "beam search", "RSA": "no", "t": "n.a.", "POS": "n.a." , "CIDEr": 0, "informativity": 0},
              {"sampling": "beam search", "RSA": "yes", "t": "no", "POS": "no" , "CIDEr": 0, "informativity": 0},
              {"sampling": "beam search", "RSA": "yes", "t": 0.5, "POS": "no" , "CIDEr": 0, "informativity": 0},
              {"sampling": "beam search", "RSA": "yes", "t": "no", "POS": "yes" , "CIDEr": 0, "informativity": 0},
              {"sampling": "beam search", "RSA": "yes", "t": 0.5, "POS": "yes" , "CIDEr": 0, "informativity": 0}]

score_df = pd.DataFrame(score_list)