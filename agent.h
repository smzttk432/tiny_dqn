#pragma once

#include "tiny_dnn/tiny_dnn.h"
#define MAXX 50
#define MAXY 50
#define GOALX 25
#define GOALY 25
#define GENMAX 500

using namespace std;

class agent {
    tiny_dnn::vec_t position;
public:
    void setposition(float x, float y);
    float dist_goal();
    void move(int action);
    tiny_dnn::vec_t now_pos();
    tiny_dnn::vec_t pos_vec();
};

void InitRand();
int random_int(int max);

float compute_reward(float bdist, float ndist, tiny_dnn::vec_t pos);

int act_by_net(tiny_dnn::vec_t result);

tiny_dnn::vec_t rewards_vec(int action, float reward);

int learn_end(tiny_dnn::vec_t pos, int gen);

int check_goal(tiny_dnn::vec_t pos, int gen);
