#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include "agent.h"
#include <Windows.h>

#include "tiny_dnn/tiny_dnn.h"
#include <sdl.h>


int main(int argc, char** argv) {
    tiny_dnn::network<tiny_dnn::sequential> mynet;
    mynet << tiny_dnn::fully_connected_layer(MAXX*MAXY, 256, false) << tiny_dnn::relu_layer()
        << tiny_dnn::fully_connected_layer(256, 4, false) << tiny_dnn::tanh_layer();
    bool loopflag = true;
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* window = SDL_CreateWindow("tiny-dnn-dqn", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 660, 660, 0);
    SDL_Renderer* render = SDL_CreateRenderer(window, -1, 0);
    SDL_Event ev;
    SDL_SetRenderDrawColor(render, 0, 0, 0, 255);
    SDL_RenderClear(render);
    SDL_SetRenderDrawColor(render, 255, 0, 0, 255);
    agent age;
    //InitRand();
    tiny_dnn::adam optim;
    int i,j;
    int action;
    float reward;
    float bdist, ndist;
    int nx, ny;
    int gen = 0;
    srand((unsigned int)time(NULL));
    float x = GOALX, y = GOALY;
    vector<tiny_dnn::vec_t> state, rewards;
    for (i = 0; i < 1000; i++) {
        //srand((unsigned int)time(NULL)*i);
        while (x == GOALX && y == GOALY) {
            x = (float)(rand() % (MAXX));
            y = (float)(rand() % (MAXY));
        }
        cout << x << "," << y << "\n";
        age.setposition(x, y);
        tiny_dnn::vec_t act;
        state.clear();
        rewards.clear();


        while (1) {
            nx = (int)age.now_pos()[0] * (600 / MAXX);
            ny = (int)age.now_pos()[1] * (600 / MAXY);
            SDL_SetRenderDrawColor(render, 0, 0, 0, 255);
            SDL_RenderClear(render);
            SDL_SetRenderDrawColor(render, 255, 255, 255, 255);
            SDL_Rect rect2 = { nx,ny,(600 / MAXX),(600 / MAXY) };
            //SDL_SetRenderDrawColor(render, 255, 255, 100, 255);
            SDL_Rect rectG = { (600 / (MAXX / GOALX)),(600 / (MAXY / GOALY)),(600 / MAXX),(600 / MAXY) };
            SDL_RenderFillRect(render, &rect2);
            SDL_RenderFillRect(render, &rectG);
            SDL_RenderPresent(render);
            gen += 1;
            if (rand() % 10 < 3) {
                action = (rand() % 4);
                cout << "random" << action << "\n";
            }
            else {
                act = mynet.predict(age.pos_vec());
                action = act_by_net(act);
                cout << "net," << act[0] << "," << act[1] << "," << act[2] << "," << act[3] << "\n";
            }
            bdist = age.dist_goal();
            state.push_back(age.pos_vec());
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
            Sleep(50);
        }
        if (check_goal(age.now_pos(), gen) == 1) {
            SDL_SetRenderDrawColor(render, 255, 0, 0, 255);
            SDL_Rect rectG = { (600 / (MAXX / GOALX)),(600 / (MAXY / GOALY)),(600 / MAXX),(600 / MAXY) };
            SDL_RenderFillRect(render, &rectG);
            SDL_RenderPresent(render);
        }
        mynet.train<tiny_dnn::mse>(optim, state, rewards, 1, 1);
    }
    mynet.save("my-network");
    SDL_DestroyRenderer(render);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
