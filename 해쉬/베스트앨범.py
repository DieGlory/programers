# from collections import defaultdict

# def solution(genres, plays):
#     music_list = defaultdict(lambda: [])
#     result = []

#     for i in zip(genres,plays,[x for x in range(len(genres))]):
#         music_list[i[0]].append([i[1],i[2]])
    
#     priority_songs = sorted(music_list.values(),key=lambda x: len(x))
#     priority_plays = sorted()


#     return result

from collections import defaultdict 
def solution(genres, plays): 
    answer = []; 
    genres_plays = defaultdict(int); 
    genres_songs = defaultdict(lambda: []) 
    i = 0; 
    for g, p in zip(genres, plays): 
        genres_plays[g] += p; 
        genres_songs[g].append((i, p)) 
        i += 1; 

    sorted_genres = sorted(genres_plays.items(), key=(lambda x: x[1]), reverse = True) 
    for g in sorted_genres: 
        sorted_g = sorted(genres_songs[g[0]], key=(lambda x: x[1]), reverse=True) 
        answer.append(sorted_g[0][0]) 
        if len(sorted_g) > 1: 
            answer.append(sorted_g[1][0]) 
    return answer


print(solution(	["classic", "pop", "classic", "classic", "pop"], [500, 600, 150, 800, 2500]))