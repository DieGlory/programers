def solution(citations):
    citations.sort(reverse=True)
    l_max = len(citations)
    v_max = citations[0]
    pivot = 0
    answer = 0
    if l_max > v_max:
        pivot = v_max
    else:
        pivot = l_max

    for i in range(pivot,0,-1):
        if citations[i-1] >= i:
            answer = i
            break
    return answer

print(solution([6,5,3,1,0]))