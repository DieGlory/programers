import heapq

def solution(scoville, K):
    answer = 0
    heap =[]
    for i in scoville:
        heapq.heappush(heap,i)
    while heap[0] < K:
        if len(heap) < 2:
            return -1
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        heapq.heappush(heap,a+b*2)
        answer+=1

    if heap[0] < K:
        return -1
    else:
        return answer


print(solution([1, 2, 3, 9, 10, 12],7))