#include <boost/python.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <cstdlib>
#include <vector>
#include <string>

using boost::multiprecision::cpp_dec_float_50;

class Action{
    // TODO: Implement Action
};

class BoardState{
    private:
        std::string board;
        int current_player;
    public:
        BoardState();
        BoardState(std::string board, int current_player);

        int getCurrentPlayer();
        std::vector<Action> getPossibleActions();
        BoardState takeAction(Action action);
        bool isTerminal();
        cpp_dec_float_50 getReward();
};