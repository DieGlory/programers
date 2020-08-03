from collections import Counter

def solution(participant, completion):
    a = (Counter(participant) - Counter(completion))
    return list(a.elements())[0]
print(solution(['leo', 'kiki', 'eden'],['eden', 'kiki']))