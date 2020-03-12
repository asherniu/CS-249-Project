import pandas as pd
import numpy as np
import pickle
author_df = pd.read_csv("data/four_area/author.txt", sep = "\t", names=["ID", "Author name"],encoding='utf8')
conf_df = pd.read_csv("data/four_area/conf.txt", sep = "\t", names=["ID", "Conference name"])
paper_df = pd.read_csv("data/four_area/paper.txt", sep = "\t", names=["ID", "Paper title"])
term_df = pd.read_csv("data/four_area/term.txt", sep = "\t", names=["ID", "Term"])
paper_author = pd.read_csv("data/four_area/paper_author.txt", sep = "\t", names=["paperID", "authorID"])
paper_conf = pd.read_csv("data/four_area/paper_conf.txt", sep = "\t", names=["paperID", "confID"])
paper_term = pd.read_csv("data/four_area/paper_term.txt", sep = "\t", names=["paperID", "termID"])
author_dict = pd.read_csv("data/DBLP_four_area/cleaned_author_dict.txt", sep = "\t", names=["ID", "Author name"], encoding='utf8')
conf_dict = pd.read_csv("data/DBLP_four_area/conf_dict.txt", sep = "\t", names=["ID", "Conference name"])
term_dict = pd.read_csv("data/DBLP_four_area/term_dict.txt", sep = "\t", names=["ID", "Term"])
author_label = pd.read_csv("data/DBLP_four_area/author_label.txt", sep = "\t", names=["authorID", "Label"])
conf_label = pd.read_csv("data/DBLP_four_area/conf_label.txt", sep = "\t", names=["confID", "Conference name", "Label"])
conf_dict_m = pd.merge(conf_dict, conf_df, on='Conference name')
author_dict_m = pd.merge(author_dict, author_df, on='Author name')
paper_conf_m = pd.merge(conf_dict_m, paper_conf, left_on='ID_y', right_on='confID')
paper_conf_m = paper_conf_m.drop(columns=['Conference name', 'ID_y','confID'])
paper_label_m = pd.merge(paper_conf_m, conf_label, left_on='ID_x', right_on='confID')
paper_label_m = paper_label_m.drop(columns=['Conference name', 'ID_x','confID'])
author_paper_label_m = pd.merge(paper_label_m,paper_author,on='paperID')
author_feature= pd.DataFrame(np.zeros(shape=(28702,4)),
                              columns=[1,2,3,4],
                              index=author_dict['ID'].unique()
                         )
for author in author_paper_label_m['authorID'].unique():
    author_dict_ID = int(author_dict_m[author_dict_m["ID_y"] == author]['ID_x'].to_string(index=False).strip())
    value_count = author_paper_label_m[author_paper_label_m['authorID'] == author]['Label'].value_counts()
    for vc in value_count.iteritems():
        label = vc[0]
        count = vc[1]
        author_feature.at[author_dict_ID, label]=count

with open('af_py2.pickle', 'wb') as f:
    pickle.dump(author_feature, f)