# WhoIsWho-IND-KDD-2024-rank6
## Reproduction Instructions
**Run ipynb 01-10 in sequence or train sh**
## Method overview 
**01 get text(title,abstract,keyword,venue) embedding  from tfidf,word2vec,chatglm3 and bge-m3 \n**
**02 For each autherID, calculate the similarity between each pid and other pids\n**
**03 Extract strongly correlated information (co author, co-org, co-keyword...)\n**
**04 tree model\n**
**05 gnn model and post-processs\n**
