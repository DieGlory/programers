from collections import deque
import heapq

def solution(jobs):
    answer = 0
    time = 0
    jobs_count = len(jobs)
    tasks = []
    jobs = deque(sorted(jobs))
    while jobs or tasks:
        while jobs and jobs[0][0] <= time:
            task = jobs.popleft()
            heapq.heappush(tasks,(task[1],task[0]))

        if tasks:
            temp = heapq.heappop(tasks)
            answer += temp[0] + time - temp[1]
            time += temp[0]
        elif jobs:
            time = jobs[0][0]

    return answer//jobs_count

print(solution([[0, 3],[0, 9], [0, 6], [1, 9], [100, 30],[100,1],[100,1],[100,2]]))
# print(solution([[0, 3], [1, 9], [2, 6]]))