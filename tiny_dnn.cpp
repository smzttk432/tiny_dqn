#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include "agent.h"

#include "tiny_dnn/tiny_dnn.h"







void run_onetime(tiny_dnn::network<tiny_dnn::sequential>& net, agent& age) {
    tiny_dnn::adam optim;
    int i;
    int action;
    float reward;
    float bdist, ndist;
    int gen = 0;
    float x = GOALX, y = GOALY;
    vector<tiny_dnn::vec_t> state, rewards;
    while (x == GOALX && y == GOALY) {
        x = (float)random_int(101);
        y = (float)random_int(101);
    }
    cout << x << "," << y << "\n";
    age.setposition(x, y);
    tiny_dnn::vec_t act;
    while (1) {
        gen += 1;
        state.clear();
        rewards.clear();
        if (random_int(10) < 3) {
            action = random_int(4);
            cout << "random" << action << "\n";
        }
        else {
            act = net.predict(age.now_pos());
            action = act_by_net(act);
            cout << "net," << act[0] << "," << act[1] << "," << act[2] << "," << act[3] << "\n";
        }
        bdist = age.dist_goal();
        state.push_back(age.now_pos());
        cout << age.now_pos()[0] << "," << age.now_pos()[1] << "-->";
        age.move(action);
        cout << age.now_pos()[0] << "," << age.now_pos()[1] << ",";
        ndist = age.dist_goal();
        reward = compute_reward(bdist, ndist, age.now_pos());
        rewards.push_back(rewards_vec(action, reward));
        cout << action << "," << reward << "\n";
        if (learn_end(age.now_pos(), gen) == 1) {
            break;
        }
    }
    net.train<tiny_dnn::mse>(optim, state, rewards, 1, 1);
}
int main() {
    tiny_dnn::network<tiny_dnn::sequential> mynet;
    mynet << tiny_dnn::fully_connected_layer(2, 256, false) << tiny_dnn::relu_layer()
        << tiny_dnn::fully_connected_layer(256, 4, false) << tiny_dnn::tanh_layer();

    agent age;
    InitRand();
    int i;
    for (i = 0; i < 1000; i++) {
        run_onetime(mynet, age);
    }
    return 0;
}
