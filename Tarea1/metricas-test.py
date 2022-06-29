#%%

relevance_query_3 = [4, 4, 3, 0, 0, 1, 3, 3, 3, 0]

k = 6
def dcg_at_k(rel, k):
    import math
    result = 0
    for i in range(k):
        discount_factor = 1/math.log(max([i+1, 2]), 2)
        gain = + (rel[i]*discount_factor)
        result = result + gain 
    return result

print(dcg_at_k(relevance_query_3, k))

#%%
def ndcg_at_k(rel, k):
    DCG = dcg_at_k(rel, k)
    IDCG = dcg_at_k(sorted(rel, reverse=True), k)
    result = DCG/IDCG
    return result

print(ndcg_at_k(relevance_query_3, k))
# %%
