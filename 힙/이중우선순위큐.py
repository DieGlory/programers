import heapq

def solution(operations):
    heap = []
    for i in operations:
        temp = i.split()
        if temp[0] == 'I':
            heapq.heappush(heap,int(temp[1]))
        elif temp[0] == 'D' and heap:
            if temp[1] == '1':
                del heap[heap.index(max(heap))]
            else:
                del heap[heap.index(min(heap))]
    if heap:
        return [max(heap),min(heap)]
    else:
        return [0,0]

print(solution(["I -45", "I 653", "D 1", "I -642", "I 45", "I 97", "D 1", "D -1", "I 333"]))