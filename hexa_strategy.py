import numpy as np

class HexaStrategyModel:
    def __init__(self, goal_level, frag_price, hexa_prob, frag_meso, reset_cost, max_level=10, max_attempts=20, iteration=100, high_value_preference=True):
        def solve_bs(x):
            cost = np.zeros((max_level + 1, max_attempts + 1), dtype=np.float32)
            go_or_stop = np.zeros_like(cost, dtype=np.int32)

            stop_cost = reset_cost + x

            # 최대 수치에 도달한 경우
            for t in range(max_level, max_attempts+1):
                if high_value_preference:
                    cost[max_level][t] = frag_price[max_level] * frag_meso * (max_attempts - t)
                else:
                    cost[max_level][t] = stop_cost
                    go_or_stop[max_level][t] = 1

            # 최대 강화 횟수에 도달한 경우
            for m in range(0, max_level+1):
                if (high_value_preference and m < goal_level) or (not high_value_preference and m > goal_level):
                    cost[m][max_attempts] = stop_cost
                    go_or_stop[m][max_attempts] = 1
                
            # 초기화 or 강화 선택
            for m in range(max_level-1, 0-1, -1):
                for t in range(max_attempts-1, m-1, -1):
                    go_cost = frag_price[m] * frag_meso + hexa_prob[m] * cost[m+1][t+1] + (1-hexa_prob[m]) * cost[m][t+1]

                    if t < 10 or (high_value_preference and m >= goal_level):
                        cost[m][t] = go_cost
                        
                    elif not high_value_preference and m > goal_level:
                        cost[m][t] = stop_cost
                        go_or_stop[m][t] = 1

                    else:
                        if go_cost <= stop_cost:
                            cost[m][t] = go_cost
                        else:
                            cost[m][t] = stop_cost
                            go_or_stop[m][t] = 1
                    
            return cost, go_or_stop

        # 이분탐색을 통해 true value 탐색
        s = 1
        e = 1000000
        ground_x = -1

        expected_cost, go_or_stop = None, None

        for i in range(iteration):
            m = (s + e) / 2
            expected_cost, go_or_stop = solve_bs(m)
            ground_x = expected_cost[0][0]

            if ground_x > m:
                s = m
            else:
                e = m

        self.expected_cost = expected_cost
        self.should_stop = go_or_stop