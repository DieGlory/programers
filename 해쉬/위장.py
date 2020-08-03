from collections import Counter

def solution(clothes):
    clothes_kind=Counter(list(map(lambda x:x[1],clothes)))
    result = 1
    for i in clothes_kind.values():
        result *= i+1
    return result -1
print(solution(	[["crow_mask", "face"], ["blue_sunglasses", "face"], ["smoky_makeup", "face"]]))