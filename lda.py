import os
import jieba
import random
import numpy
def read_document(doc_num,stop_words,dic):
    doc = []
    for m in range(doc_num):
        doc.append([])
        path = os.getcwd() + "\\data\\1"+str(m)+".txt"
        f = open(path)
        try:
            ftext = f.read()

        finally:
            f.close()  
        
        
        f_seg_list = jieba.cut(ftext)        
        term_id = 0
    
        for myword in f_seg_list:
        
            if not(myword.strip() in stop_words)and len(myword.strip())>1:
                dic.setdefault(myword,0)
               
                doc[m].append(myword)
        for key in dic.keys():
            dic[key] = term_id
            term_id += 1
    return doc,dic
        
        

            
  
    
def load_stopwords():
    f_stop = open('stopwords.txt')
    try:
        f_stop_text = f_stop.read()

    finally:
        f_stop.close()

    f_stop_seg_list = f_stop_text.split('\n')
    return f_stop_seg_list



def gibbs_sampling(z, m, i, nt, nd, nt_sum, nd_sum, term):
    topic = z[m][i]      #当前主题
    nt[term][topic] -= 1 #去除当前词
    nd[m][topic] -= 1    
    nt_sum[topic] -= 1
    nd_sum[m] -= 1

    topic_alpha = topic_number * alpha
    term_beta = len(dic) * beta
    p = [0 for x in range(topic_number)]   #p[k]: 属于主题k的概率
    for k in range(topic_number):
        p[k] = (nd[m][k] + alpha) / (nd_sum[m] + topic_alpha) \
                * (nt[term][k] + beta) / (nt_sum[k] +term_beta)
        if k >= 1:             #转换成累加概率
            p[k] += p[k-1]
    gs = random.random() * p[topic_number-1]  #随机采样
    new_topic = 0
    while new_topic < topic_number:
        if p[new_topic] > gs:
            break
        new_topic += 1
    nt[term][new_topic] += 1
    nd[m][new_topic] += 1
    nt_sum[new_topic] += 1
    nd_sum[m] += 1
    z[m][i] = new_topic #新主题   
    
def calc_theta(nd, nd_sum): #每个文档的主题分布
    doc_num = len(nd)
    topic_alpha = topic_number * alpha
    theta = [[0 for t in range(topic_number)] for d in range(doc_num)]
    for m in range(doc_num):
        for k in range(topic_number):
            theta[m][k] = (nd[m][k] + alpha) / (nd_sum[m] + topic_alpha)
    return theta

def calc_phi(nt, nt_sum):  #每个主题的词分布
    term_num = len(nt)
    term_beta = term_num * beta
    phi = [[0 for w in range(term_num)] for t in range(topic_number)]
    for k in range(topic_number):
        for term in range(term_num):
            phi[k][term] = (nt[term][k] + beta) / (nt_sum[k] + term_beta)
    return phi

def lda(z, nt, nd, nt_sum, dic, doc):
    doc_num = len(z)    #文档数
    for time in range(50):
        for m in range(doc_num):
            doc_length = len(z[m])  #第m篇文档的长度
            for i in range(doc_length):
                term = dic[doc[m][i]]  #第m篇文档的第i个词 --> 词汇
                gibbs_sampling(z, m, i, nt, nd, nt_sum, nd_sum, term)
        theta = calc_theta(nd, nd_sum)   #计算每个文档的主题分布
        phi = calc_phi(nt, nt_sum)       #计算每个主题的的词分布
        return theta,phi


                
def init_topic(doc, nt, nd, nt_sum, nd_sum, dic):
    #随机分配类型
    
    #topic_number = len(nd[m]) 

    doc_num = len(nd)
    z = [[0 for x in range(len(doc[m]))] for m in range(doc_num)]
    for x in range(doc_num):
        nd_sum[x] = len(doc[x])
        for y in range(len(doc[x])):
            topic = random.randint(0,topic_number-1)
            z[x][y] = topic
            term = dic.get(doc[x][y])
            nt[term][topic] += 1
            nd[x][topic] += 1
            nt_sum[topic] += 1

    return z                

def show_result(theta, phi, dic):
    term_num = len(dic)
    doc_num = len(theta)
    topic_num = len(phi)
    dics = []
    dics = list(dic.items())
    #print(dics)
    dics.sort(key = lambda i:i[1],reverse = False)
    #print(dics)
    #排序后的词元组
    
    topic_word=[]
    for t in range(topic_num):
        topic_word.append([])
        print("主题" +str(t)+ ": ",end = " ")
        top_terms = []
        top_terms = [(n,phi[t][n]) for n in range(term_num)]
        top_terms.sort(key = lambda i:i[1],reverse = True)
        #排序后最优主题词汇
        for top in range(top_num):
            top_word = dics[top_terms[top][0]][0]
            topic_word[t].append(top_word)
            #top_word = sorted_dic({value:key for key, value in dic.items()})[top_terms[top][0]]
            print(top_word +"(" + str(top_terms[top][1]) +") ",end = "")
            
            
        print("\n")
    doc_topic = []
    for m in range(doc_num):
        print("文档"+str(m)+": ",end = "")
        
        doc_topic.append([])
        doc_topic[m] = [(k,theta[m][k]) for k in range(topic_num)]
        doc_topic[m].sort(key = lambda i:i[1],reverse = True)
       
        print(doc_topic[m][0],end = " ")
        print(topic_word[m],end = " ")
        
        print("\n")
    #print(str(dics[top_terms[0:top_num][0]][0]))        
    print(topic_word) 
    print(doc_topic)
        

if __name__ == "__main__":
    doc_num = 3 #文档数目
    #载入停用词表
    stop_words = load_stopwords()
    dic = {}
    doc,dic = read_document(doc_num,stop_words,dic)
    print(len(doc))
    print(len(dic))
    #LDA
    
 
    topic_number = 20  #主题数
    term_num = len(dic)        #词汇数
    top_num = 3                #最优主题数
    alpha = 50/topic_number
    beta = 0.01
    # nt[w][t]：第term个词属于第t个主题的次数
    nt = [[0 for t in range(topic_number)] for term in range(term_num)]
    # nd[d][t]: 第d个文档中出现第t个主题的次数
    nd = [[0 for t in range(topic_number)] for d in range(doc_num)]
    # nt_sum[t]: 第t个主题出现的次数(nt矩阵的第t列)
    nt_sum = [0 for t in range(topic_number)]
    # nd_sum[d]: 第d个文档的长度（nd矩阵的第d行）
    nd_sum = [0 for d in range(doc_num)]
    
    z = init_topic(doc, nt, nd, nt_sum, nd_sum, dic)
    theta, phi = lda(z, nt, nd, nt_sum, dic, doc)
    # 输出每个文档的主题和每个主题的关键字
    show_result(theta, phi, dic)