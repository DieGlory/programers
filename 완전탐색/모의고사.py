def solution(answers):
    scores = []
    answer = []
    length = len(answers)
    solver= [[1,2,3,4,5],[2,1,2,3,2,4,2,5],[3,3,1,1,2,2,4,4,5,5]]
    check = lambda a,b : a==b
    solver_a = []

    for i in range(3):
        solver_a.append(solver[i] * (length//len(solver[i])) + solver[i][:(length%len(solver[i]))])
    for i in range(3):
        scores.append(sum(list(map(check,answers, solver_a[i]))))

    max_score = max(scores)
    for i in range(3):
        if scores[i] == max_score:
            answer.append(i+1)
    return answer

print(solution([1,2,3,4,5]))