import copy
import random

from game.ai_angent import AIAgent
import tensorflow as tf
import numpy as np

action_list = {0: [0, -1], 1: [0, 1], 2: [-1, 0], 3: [1, 0], 4: [-1, -1], 5: [1, -1], 6: [-1, 1], 7: [1, 1]}

MAP_LENGTH = 14

random.seed(1)

def get_game_map():
    game_map = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                [1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1],
                [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1],
                [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

    return game_map


def reset_observation(game_map, hero, enemy, daoju_list, safe_list):
    observation = np.zeros((14, 14, 5))

    for i in range(MAP_LENGTH):
        for j in range(MAP_LENGTH):
            if game_map[i][j] == 1:
                observation[j][i][2] = 1.0

    observation[hero[0]][hero[1]][0] = 1.0

    observation[enemy[0]][enemy[1]][1] = 1.0

    for daoju in daoju_list:
        observation[daoju[0]][daoju[1]][3] = 1.0

    for safe in safe_list:
        observation[safe[0]][safe[1]][4] = 1.0

    return observation


def get_reward(action, observation, hero, enemy, daoju_list, safe_list):
    _action = action_list[action]

    new_pos = [hero[0] + _action[0], hero[1] + _action[1]]

    if observation[new_pos[0]][new_pos[1]][2] == 1.0 or new_pos == enemy:
        return -1.0, True

    if new_pos in safe_list:
        return 1.0, True

    if new_pos == [enemy[0] + 1, enemy[1]] or new_pos == [enemy[0] - 1, enemy[1]] or \
            new_pos == [enemy[0], enemy[1] + 1] or new_pos == [enemy[0], enemy[1] - 1]:
        return 0.2, False

    if new_pos in daoju_list:
        return 0.2, False

    return -0.002, False


def get_new_observation(action, hero, daoju_list, observation):
    _observation = copy.deepcopy(observation)
    _action = action_list[action]

    new_pos = [hero[0] + _action[0], hero[1] + _action[1]]

    # old position change to wall
    _observation[hero[0]][hero[1]][2] = 1.0

    _observation[hero[0]][hero[1]][0] = 0.0

    _observation[new_pos[0]][new_pos[1]][0] = 1.0

    return new_pos, _observation


def get_hero_enemy_pool(game_map):
    hero_enemy_pool = []
    for i in range(len(game_map)):
        for j in range(len(game_map[0])):
            hero = [j, i]
            if game_map[i][j] == 1:
                continue
            for i_ in range(len(game_map)):
                for j_ in range(len(game_map[0])):
                    enemy = [j_, i_]
                    if game_map[i_][j_] == 1 or hero == enemy:
                        continue

                    hero_enemy_pool.append([hero, enemy])

    return hero_enemy_pool


def get_safe_list(game_map):
    safe_list = []
    for i in range(1, len(game_map) - 1):
        for j in range(1, len(game_map[0]) - 1):
            count = game_map[i - 1][j] + game_map[i + 1][j] + game_map[i][j - 1] + game_map[i][j + 1]

            if count >= 3:
                safe_list.append([j, i])

    return safe_list


def get_daoju_list(game_map, hero, enemy, daoju):
    while True:
        first_daoju = [random.randint(1, MAP_LENGTH - 1), random.randint(1, MAP_LENGTH - 1)]

        if game_map[first_daoju[1]][first_daoju[0]] == 1 or first_daoju == hero or first_daoju == enemy or \
                daoju == first_daoju:
            continue

        return first_daoju


def run_this():
    game_map = get_game_map()

    safe_list = get_safe_list(game_map)

    pool = get_hero_enemy_pool(game_map)

    random.shuffle(pool)

    sess = tf.Session()

    ai_agent = AIAgent(sess, 8, tf_device='/gpu:*')
    # ai_agent = AIAgent(sess, 8, tf_device='/cpu:*', epsilon_eval=0.0, eval_mode=True)

    sess.run(tf.global_variables_initializer())

    # writer = tf.summary.FileWriter("/tmp/tensor_logs/", sess.graph)

    ai_agent.unbundle('checkpoint', 4609, {})

    pool_len = len(pool)

    print("pool lenght is " + str(pool_len))

    for index in range(4608, pool_len):
        # random_index = random.randint(0, pool_len)
        hero, enemy = pool[index]

        print("index is " + str(index) + ". hero is " + str(hero) + ". enemy is" + str(enemy))

        first_daoju = get_daoju_list(game_map, hero, enemy, [0, 0])
        second_daoju = get_daoju_list(game_map, hero, enemy, first_daoju)

        daoju_list = [first_daoju, second_daoju]

        # print("daoju list is " + str(daoju_list))

        _safe_list = []

        for safe in safe_list:
            if safe != hero and safe != enemy:
                _safe_list.append(safe)

        over = False
        reward_count = 0
        step = 0
        while step < 5000:
            if over:
                break

            _hero = hero

            observation = reset_observation(game_map, _hero, enemy, daoju_list, _safe_list)
            action = ai_agent.begin_episode(observation)

            # move_list = [_hero]
            while True:
                step += 1
                reward, done = get_reward(action, observation, _hero, enemy, daoju_list, _safe_list)

                _hero, _observation = get_new_observation(action, _hero, daoju_list, observation)
                observation = _observation
                # move_list.append(_hero)

                if done:
                    # print(str(move_list))

                    ai_agent.end_episode(reward)

                    if reward == 1.0:
                        reward_count += 1

                    if reward_count > 2:
                        over = True
                    break
                else:
                    action = ai_agent.step(reward, observation)

        ai_agent.bundle_and_checkpoint('checkpoint', index)

    ai_agent.bundle_and_checkpoint('checkpoint', pool_len)

    sess.close()


if __name__ == '__main__':
    run_this()
