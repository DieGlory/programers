def solution(citations):
    # 정답 초기화
    # answer = H-Index
    answer = 0

    # n : 총 논문 수
    n = len(citations)

    # h 값들을
    list_h = []

    # 내림 정렬
    # 각 원소에 대해서 그 원소 값 이상 인용된 논문 수는 자신의 index+1편 만큼 있음
    citations.sort(reverse=True)

    # [99, 85, 73, 22, 11, 5]

    # index+1이 h 값
    for i in range(n):
        if citations[i] >= i + 1:
            list_h.append(i + 1)

    answer = list_h[-1]

    return answer
print(solution([]))