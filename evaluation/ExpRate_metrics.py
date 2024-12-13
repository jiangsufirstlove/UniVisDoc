import numpy


def cal_distance(word1, word2):
    m = len(word1)
    n = len(word2)
    if m*n == 0:
        return m+n
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            a = dp[i-1][j] + 1
            b = dp[i][j-1] + 1
            c = dp[i-1][j-1]
            if word1[i-1] != word2[j-1]:
                c += 1
            dp[i][j] = min(a, b, c)
    return dp[m][n]


def compute_edit_distance(prediction, label):
    prediction = prediction.strip().split(' ')
    label = label.strip().split(' ')
    distance = cal_distance(prediction, label)
    return distance


def compute_exprate_1(predictions, references):   #输入带空格
    total_line = 0
    total_line_rec = 0
    total_line_error_1 = 0
    total_line_error_2 = 0 
    for i in range(len(references)):
        pre = predictions[i]
        ref = references[i]
        dist = compute_edit_distance(pre, ref)
        total_line += 1
        if dist == 0:
            total_line_rec += 1
        elif dist ==1:
            total_line_error_1 +=1
        elif dist ==2:
            total_line_error_2 +=1
    exprate = float(total_line_rec)/total_line
    error_1 = float(
        total_line_error_1 + total_line_rec
    )/total_line
    error_2 = float(
        total_line_error_2 + total_line_error_1 +total_line_rec
    )/total_line

    return round(exprate*100, 2),round(error_1*100, 2),round(error_2*100, 2)











def cmp_result(label,rec):
    dist_mat = numpy.zeros((len(label)+1, len(rec)+1),dtype='int32')
    dist_mat[0,:] = range(len(rec) + 1)
    dist_mat[:,0] = range(len(label) + 1)
    for i in range(1, len(label) + 1):
        for j in range(1, len(rec) + 1):
            hit_score = dist_mat[i-1, j-1] + (label[i-1] != rec[j-1])
            ins_score = dist_mat[i,j-1] + 1
            del_score = dist_mat[i-1, j] + 1
            dist_mat[i,j] = min(hit_score, ins_score, del_score)
    dist = dist_mat[len(label), len(rec)]
    return dist, len(label)

def compute_exprate(predictions, references):  #输入不带空格
    total_label = 0
    total_line = 0
    total_line_rec = 0
    total_line_error_1 = 0
    total_line_error_2 = 0 
    for i in range(len(references)):
        pre = predictions[i]#.split()
        ref = references[i]#.split()
        #print(pre, ref)
        dist, llen = cmp_result(pre, ref)
        total_label += llen
        total_line += 1
        if dist == 0:
            total_line_rec += 1
        elif dist ==1:
            total_line_error_1 +=1
        elif dist ==2:
            total_line_error_2 +=1
    exprate = float(total_line_rec)/total_line
    error_1 = float(
        total_line_error_1 + total_line_rec
    )/total_line
    error_2 = float(
        total_line_error_2 + total_line_error_1 +total_line_rec
    )/total_line

    return round(exprate*100, 2),round(error_1*100, 2),round(error_2*100, 2)