relevance_query_1 = [1,0,1,1,1,1,1,0,1,0,1,0,0,1,0,0,0,0,0,1]
k = 3

def precision(relevance_query):
    n = len(relevance_query)
    x = 0
    for item in relevance_query:
        if(item == 1):
            x += 1
    return x/n

def precision_at_k(relevance_query,k):
    x = 0
    if k > 0 and k <= len(relevance_query):
        for i in range(0, k):
            if(relevance_query[i] == 1):
                x += 1
        return x/k
    else:
        return None

print(precision(relevance_query_1))
print(precision_at_k(relevance_query_1,k))