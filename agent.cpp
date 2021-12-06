#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include "agent.h"

#include "tiny_dnn/tiny_dnn.h"

void agent::setposition(float x, float y) {
    this->position = { x,y };
}
float agent::dist_goal() {
    return sqrt(pow(this->position[0] - GOALX, 2) + pow(this->position[1] - GOALY, 2));
}
void agent::move(int action) {
    switch (action) {
    case 0:
        this->position[0] += 1;
        break;
    case 1:
        this->position[0] -= 1;
        break;
    case 2:
        this->position[1] += 1;
        break;
    case 3:
        this->position[1] -= 1;
        break;
    }
}

tiny_dnn::vec_t agent::now_pos() {
    return this->position;
}

void InitRand()
{
    srand((unsigned int)time(NULL));
}

int random_int(int max) {
    return rand() % max;
}

float compute_reward(float bdist, float ndist, tiny_dnn::vec_t pos) {
    if (ndist == 0) {
        return 1.0;
    }
    else if (pos[0] < 0 || pos[1] < 0 || pos[0]>99 || pos[1]>99) {
        return -1.0;
    }
    else {
        return bdist - ndist;
    }
}

int act_by_net(tiny_dnn::vec_t result) {
    float max = -2;
    int i, a = 0;
    for (i = 0; i < 4; i++) {
        if (result[i] > max) {
            a = i;
            max = result[i];
        }
    }
    return a;
}

tiny_dnn::vec_t rewards_vec(int action, float reward) {
    tiny_dnn::vec_t rewards = { 0,0,0,0 };
    rewards[action] = reward;
    return rewards;
}

int learn_end(tiny_dnn::vec_t pos, int gen) {
    if (pos[0] < 0 || pos[1] < 0 || gen == GENMAX || pos[0]>100 || pos[1]>100 || (pos[0] == GOALX && pos[1] == GOALY)) {
        return 1;
    }
    else {
        return 0;
    }
}