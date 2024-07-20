import streamlit as st
import numpy as np
import pandas as pd

import copy
from hexatable import hexa_table, hexa_table_frag, hexa_frag_cost, hex_frag_multiplier, hexa_prob
from hexa_strategy import HexaStrategyModel

from linalg import *

st.set_page_config(layout="wide")

MAX_LEVEL = 10
MAX_ATTEMPTS = 20
ITERATIONS = 100
INF = 1 << 64

# CSS 파일 로드
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# CSS 파일 경로
css_file = 'static/style.css'
load_css(css_file)


# 페이지 레이아웃 설정
if 'generated' not in st.session_state:
    st.session_state['generated'] = False

if st.session_state['generated']:
    factors, _, statistics = st.columns([1, 0.1, 2])
else:
    _, factors, _ = st.columns([1, 2, 1])

# 왼쪽 열에 입력 폼 배치
with factors:
    with st.container(border=True):
        st.title('헥사스탯 돌깎기 계산기')

        frag_price = st.number_input('조각 가격 (단위: 억 메소)', min_value=0.0, value=0.07, format="%.3f", step=0.0)
        reset_cost = st.number_input('초기화 비용 (단위: 억 메소)', value=0.1, disabled=True)
        
        col3, col4 = st.columns([3, 1])
        with col3:
            goal_level = st.number_input('목표 메인스탯 레벨', min_value=0, value=10, max_value=MAX_LEVEL, step=1)
        with col4:
            st.write('')
            if 'high_toggle_state' not in st.session_state:
                st.session_state['high_toggle_state'] = True
            
            st.markdown('<div sty>', unsafe_allow_html=True)
            if st.button('이상' if st.session_state['high_toggle_state'] else '이하'):
                st.session_state['high_toggle_state'] = not st.session_state['high_toggle_state']
                st.rerun()

        if 'frag_toggle_state' not in st.session_state:
            st.session_state['frag_toggle_state'] = True

        col5, col6 = st.columns([3, 1])
        with col5:
            if st.session_state['frag_toggle_state']:
                num_frags = st.number_input('조각 개수', min_value=0, value=1000, step=10*hex_frag_multiplier)
            else:
                num_frags = st.number_input('조각 개수', disabled=True)  # 멸망전 선택 시 조각 개수 무시
        with col6:
            st.write('')
            st.markdown('<div sty>', unsafe_allow_html=True)
            if st.button('조각 개수' if st.session_state['frag_toggle_state'] else '멸망전'):
                st.session_state['frag_toggle_state'] = not st.session_state['frag_toggle_state']
                st.rerun()

        sunday_maple = st.checkbox('썬데이 메이플 적용')

        if st.button('깎기'):
            high_value_preference = st.session_state['high_toggle_state']
            st.session_state['high_value_preference'] = high_value_preference
            st.session_state['goal_level'] = goal_level
            st.session_state['frag_mode'] = st.session_state['frag_toggle_state']

            hsm = HexaStrategyModel(
                goal_level, hexa_frag_cost, hexa_prob[sunday_maple], 
                frag_price, reset_cost, 
                MAX_LEVEL, MAX_ATTEMPTS, ITERATIONS, 
                high_value_preference
            )

            ######## Apply Strategy ########

            # 조각 소모
            if st.session_state['frag_mode']:
                st.session_state['num_frags'] = num_frags
                table = hexa_table_frag[sunday_maple]
                strategy_table = copy.deepcopy(table)

                x = {(0, 0, 0): 1} # main-level, attempts, spent-frags
                init_states = []
                for fsstate, prob in strategy_table.transitions.items():
                    from_state, to_state = fsstate
                    m, t, f = to_state
                    if hsm.should_stop[m][t] and f == 0:
                        init_states.append((from_state, to_state, prob))

                for from_state, to_state, prob in init_states:
                    del strategy_table.transitions[(from_state, to_state)]
                    prev_prob = strategy_table.transitions.get((from_state, (0, 0, 0)), 0)
                    strategy_table.add_transition(from_state, (0, 0, 0), prev_prob + prob)


                if high_value_preference:
                    for level in range(goal_level, MAX_LEVEL+1):
                        strategy_table.clear_transition((level, MAX_ATTEMPTS, 0))
                        strategy_table.add_transition((level, MAX_ATTEMPTS, 0), (level, MAX_ATTEMPTS, 0), 1.0)   
                else:
                    for level in range(0, goal_level+1):
                        strategy_table.clear_transition((level, MAX_ATTEMPTS, 0))
                        strategy_table.add_transition((level, MAX_ATTEMPTS, 0), (level, MAX_ATTEMPTS, 0), 1.0)

                strategy_table.compile()

                ######## Do Reinforce ########

                result_vector = np.zeros(MAX_LEVEL+1)

                if high_value_preference:
                    result = reinforce(strategy_table, x, num_frags // hex_frag_multiplier)

                    for i in range(MAX_LEVEL+1):
                        result_vector[i] = result[(i,)]

                else:
                    result = reinforce(strategy_table, x, num_frags // hex_frag_multiplier, False)
                    result = strategy_table.compress_state_dict(strategy_table.vector_to_state_dict(result), [0, 1])
                    for i in range(MAX_LEVEL+1):
                        result_vector[i] = result.get((i, MAX_ATTEMPTS), 0)

                st.session_state['result_vector'] = result_vector

            # 조각 무한 멸망전
            else:
                table = hexa_table[sunday_maple]
                strategy_table = copy.deepcopy(table)

                x = {(0, 0): 1}  # main-level, attempts

                for m in range(MAX_LEVEL+1):
                    for t in range(MAX_ATTEMPTS+1):
                        if hsm.should_stop[m][t]:
                            strategy_table.clear_transition((m, t))
                            strategy_table.add_transition((m, t), (0, 0), 1.0)

                if high_value_preference:
                    for level in range(goal_level, MAX_LEVEL+1):
                        strategy_table.clear_transition((level, MAX_ATTEMPTS))
                        strategy_table.add_transition((level, MAX_ATTEMPTS), (level, MAX_ATTEMPTS), 1.0)
                else:
                    for level in range(0, goal_level+1):
                        strategy_table.clear_transition((level, MAX_ATTEMPTS))
                        strategy_table.add_transition((level, MAX_ATTEMPTS), (level, MAX_ATTEMPTS), 1.0)

                strategy_table.compile()

                ######## Calculate Expectation ########

                def cost_branch(state, hsm, only_frag=False):
                    i, j = state
                    if hsm.should_stop[i][j]:
                        return reset_cost if not only_frag else 0
                    elif j == 20:
                        return 0
                    else:
                        return hexa_frag_cost[i] * hex_frag_multiplier * frag_price if not only_frag else hexa_frag_cost[i] * hex_frag_multiplier

                strategy_cost = strategy_table.create_vector({
                    (i, j): cost_branch((i, j), hsm) for i, j in strategy_table.states
                })

                strategy_frag_cost = strategy_table.create_vector({
                    (i, j): cost_branch((i, j), hsm, True) for i, j in strategy_table.states
                })

                xvector = strategy_table.create_vector(x)
                migs = matrix_semi_inf_geometric_series(strategy_table.get_matrix())

                expected_cost = xvector @ migs @ strategy_cost.T
                expected_frag_cost = xvector @ migs @ strategy_frag_cost.T

                st.session_state['expected_cost'] = expected_cost
                st.session_state['expected_frag_cost'] = expected_frag_cost

                ######## Do Reinforce ########

                result = reinforce(strategy_table, x, INF)

                result_vector = np.zeros(MAX_LEVEL+1)
                for i in range(MAX_LEVEL+1):
                    result_vector[i] = result[(i,)]

                st.session_state['result_vector'] = result_vector

            ######## Show Strategy ########

            last_one_indices = []
            for col in range(hsm.should_stop.shape[1]):
                rows_with_ones = np.where(hsm.should_stop[:, col] == 1)[0]
                if rows_with_ones.size > 0:
                    last_one_indices.append(rows_with_ones[-1])
                else:
                    last_one_indices.append(-1)  # 1이 없는 경우 -1로 표시

            prev = -1
            data = {'강화 횟수': [], '메인 스탯': []}
            for i in range(10, MAX_ATTEMPTS+1):
                if last_one_indices[i] != prev:
                    data['강화 횟수'].append(f'{i}회 강화')
                    data['메인 스탯'].append(f'메인스탯 {last_one_indices[i]} 이하면 초기화')
                prev = last_one_indices[i]

            df = pd.DataFrame(data)
            st.session_state['df'] = df

            st.session_state['generated'] = True
            st.rerun()

if st.session_state['generated']:
    # 오른쪽 열에 테이블과 그래프 표시
    with statistics:
        with st.container(border=True):
            if 'df' in st.session_state and 'result_vector' in st.session_state:
                goal_level = st.session_state['goal_level']
                high_value_preference = st.session_state['high_value_preference']
                result_vector = st.session_state['result_vector']

                if not st.session_state['frag_mode'] and 'expected_cost' in st.session_state and 'expected_frag_cost' in st.session_state:
                    expected_cost = st.session_state['expected_cost']
                    expected_frag_cost = st.session_state['expected_frag_cost']

                    st.write(f'### 메인스탯 {goal_level} {"이상" if high_value_preference else "이하"} 멸망전')
                    st.write(f'평균 메소 소모량: {round(expected_cost, 2)}억 메소 / 평균 조각 소모량: {round(expected_frag_cost, 2)}개')

                elif st.session_state['frag_mode'] and 'num_frags' in st.session_state:
                    num_frags = st.session_state['num_frags'] 
                    success = np.sum(result_vector[goal_level:]) if high_value_preference else np.sum(result_vector[:goal_level+1])
                    st.write(f'### 메인스탯 {goal_level} {"이상" if high_value_preference else "이하"} 도전 - 조각 {num_frags}개 달성률: {round(success * 100, 2)}%')

                # result_vector를 st.bar_chart로 시각화
                chart_data = pd.DataFrame({
                    '메인 레벨': range(MAX_LEVEL + 1),
                    '결과 값': result_vector
                }).set_index('메인 레벨')
                st.bar_chart(chart_data)

                st.write("### 전략 생성 결과")
                st.table(st.session_state['df'])
