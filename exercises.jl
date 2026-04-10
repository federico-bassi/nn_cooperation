Pairs = [1 3 5; 
        2 4 6]
IA = [1, 0 , 2, 1, 0, 2]
t_pre = IA[Pairs]
t_pre = vec(sum(t_pre .== 0, dims = 1))

A = [10, 20, 30, 40]
B = [1, 3, 4]

n=10